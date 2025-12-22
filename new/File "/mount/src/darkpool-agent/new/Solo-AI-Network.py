import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { 
  Settings as SettingsIcon, 
  Users, 
  User, 
  Info, 
  Send, 
  FileText, 
  BrainCircuit, 
  Loader2, 
  LayoutDashboard, 
  ShieldCheck, 
  Zap, 
  Target, 
  LifeBuoy, 
  Search, 
  Image as ImageIcon, 
  ExternalLink, 
  Upload, 
  Mic, 
  MicOff, 
  Volume2, 
  Square, 
  Menu, 
  X, 
  Lock, 
  ServerCrash,
  ChevronRight,
  Sparkles,
  Command
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { ANALYSTS } from './constants';
import { AppState, CSOMode, TierLevel, GroundingChunk } from './types';
import { GeminiService } from './services/geminiService';

const TIER_INFO: Record<TierLevel, { name: string; color: string; icon: any; bg: string }> = {
  1: { name: 'Sovereign', color: 'text-rose-400', bg: 'bg-rose-500/10', icon: ShieldCheck },
  2: { name: 'Strategic', color: 'text-indigo-400', bg: 'bg-indigo-500/10', icon: Target },
  3: { name: 'Execution', color: 'text-emerald-400', bg: 'bg-emerald-500/10', icon: Zap },
  4: { name: 'Support', color: 'text-amber-400', bg: 'bg-amber-500/10', icon: LifeBuoy },
};

const MODES = [
  { name: 'Build', keys: ['a3', 'a4', 'a7', 'a9', 'a2'], color: 'from-emerald-500/20 to-indigo-500/20' },
  { name: 'Audit', keys: ['a1', 'a11', 'a14', 'a18', 'a19'], color: 'from-rose-500/20 to-slate-500/20' },
  { name: 'Growth', keys: ['a5', 'a8', 'a12', 'a13', 'a2'], color: 'from-amber-500/20 to-emerald-500/20' },
];

const App: React.FC = () => {
  const isApiKeyMissing = !process.env.API_KEY || process.env.API_KEY === 'undefined';
  
  const [activeTab, setActiveTab] = useState<'council' | 'single' | 'live' | 'lab'>('council');
  const [selectedAnalystKey, setSelectedAnalystKey] = useState<string>('a2');
  const [isLoading, setIsLoading] = useState(false);
  const [isBriefing, setIsBriefing] = useState(false);
  const [userInput, setUserInput] = useState('');
  const [selectedCouncilKeys, setSelectedCouncilKeys] = useState<string[]>(['a1', 'a2', 'a5', 'a11']);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(!process.env.APP_PASSWORD);
  const [passwordInput, setPasswordInput] = useState('');

  const [state, setState] = useState<AppState>({
    plan: null,
    lastUserPrompt: '',
    lastCouncilResponses: {},
    lastCouncilGrounding: {},
    lastDecisionCard: '',
    chatLogs: Object.fromEntries(ANALYSTS.map(a => [a.key, []])),
    memoryNotes: Object.fromEntries(ANALYSTS.map(a => [a.key, ''])),
    csoMode: CSOMode.Critique,
    settings: {
      model: 'gemini-3-pro-preview',
      temperature: 0.3,
      maxTokens: 4096,
      thinkingBudget: 16000,
      enableSearch: false
    },
    currentImage: null,
    editedImage: null
  });

  const [isLiveActive, setIsLiveActive] = useState(false);
  const liveSessionRef = useRef<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (activeTab === 'single') {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [state.chatLogs, activeTab]);

  const groupedAnalysts = useMemo(() => {
    const groups: Record<TierLevel, typeof ANALYSTS> = { 1: [], 2: [], 3: [], 4: [] };
    ANALYSTS.forEach(a => groups[a.tier].push(a));
    return groups;
  }, []);

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (passwordInput === process.env.APP_PASSWORD) {
      setIsAuthenticated(true);
    } else {
      alert("Invalid Access Code");
    }
  };

  // Implement mandatory encode helper as per guidelines
  const encode = (bytes: Uint8Array) => {
    let binary = '';
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  };

  // Implement mandatory decode helper as per guidelines
  const decode = (base64: string) => {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
  };

  const decodeAudioData = async (data: Uint8Array, ctx: AudioContext, sampleRate: number, numChannels: number): Promise<AudioBuffer> => {
    const dataInt16 = new Int16Array(data.buffer);
    const frameCount = dataInt16.length / numChannels;
    const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);
    for (let channel = 0; channel < numChannels; channel++) {
      const channelData = buffer.getChannelData(channel);
      for (let i = 0; i < frameCount; i++) {
        channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
      }
    }
    return buffer;
  };

  // Fix for error: Added missing toggleCouncilRole function
  const toggleCouncilRole = (key: string) => {
    setSelectedCouncilKeys(prev => 
      prev.includes(key) 
        ? prev.filter(k => k !== key) 
        : [...prev, key]
    );
  };

  const playBriefing = async () => {
    if (!state.lastDecisionCard || isBriefing) return;
    setIsBriefing(true);
    const gemini = new GeminiService();
    try {
      const audioBytes = await gemini.generateTTS(state.lastDecisionCard);
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      const buffer = await decodeAudioData(audioBytes, ctx, 24000, 1);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.onended = () => setIsBriefing(false);
      source.start();
    } catch (err) {
      console.error(err);
      setIsBriefing(false);
    }
  };

  const startLiveWarRoom = async () => {
    if (isLiveActive) {
      liveSessionRef.current?.close();
      setIsLiveActive(false);
      return;
    }
    setIsLiveActive(true);
    const gemini = new GeminiService();
    audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
    const outputNode = audioContextRef.current.createGain();
    outputNode.connect(audioContextRef.current.destination);
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const inputCtx = new AudioContext({ sampleRate: 16000 });
    const sessionPromise = gemini.connectLive({
      onopen: () => {
        const source = inputCtx.createMediaStreamSource(stream);
        const scriptProcessor = inputCtx.createScriptProcessor(4096, 1, 1);
        scriptProcessor.onaudioprocess = (e) => {
          const inputData = e.inputBuffer.getChannelData(0);
          const int16 = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) int16[i] = inputData[i] * 32768;
          // Fix: Use implemented encode helper for Live API compliance
          const base64 = encode(new Uint8Array(int16.buffer));
          sessionPromise.then(s => s.sendRealtimeInput({ media: { data: base64, mimeType: 'audio/pcm;rate=16000' } }));
        };
        source.connect(scriptProcessor);
        scriptProcessor.connect(inputCtx.destination);
      },
      onmessage: async (msg) => {
        const audioData = msg.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
        if (audioData && audioContextRef.current) {
          // Fix: Use implemented decode helper for Live API compliance
          const bytes = decode(audioData);
          const buffer = await decodeAudioData(bytes, audioContextRef.current, 24000, 1);
          const source = audioContextRef.current.createBufferSource();
          source.buffer = buffer;
          source.connect(outputNode);
          nextStartTimeRef.current = Math.max(nextStartTimeRef.current, audioContextRef.current.currentTime);
          source.start(nextStartTimeRef.current);
          nextStartTimeRef.current += buffer.duration;
        }
        if (msg.serverContent?.interrupted) nextStartTimeRef.current = 0;
      },
      onerror: (e) => console.error("Live Error", e),
      onclose: () => setIsLiveActive(false)
    }, "You are the Council Chair in a Live War Room. Facilitate high-stakes strategic sessions. Keep responses concise and fast.");
    liveSessionRef.current = await sessionPromise;
  };

  const runCouncil = async () => {
    if (!userInput.trim()) return;
    setIsLoading(true);
    const gemini = new GeminiService();
    const responses: Record<string, string> = {};
    const groundings: Record<string, GroundingChunk[]> = {};
    try {
      await Promise.all(selectedCouncilKeys.map(async (key) => {
        const a = ANALYSTS.find(x => x.key === key)!;
        const out = await gemini.generate(a.systemPrompt, userInput, [], state.settings);
        responses[key] = out.text;
        groundings[key] = out.sources;
      }));
      const analystNames: Record<string, string> = {};
      ANALYSTS.forEach(a => analystNames[a.key] = a.name);
      const decisionCard = await gemini.synthesize(userInput, responses, analystNames, state.settings);
      setState(prev => ({
        ...prev,
        lastUserPrompt: userInput,
        lastCouncilResponses: responses,
        lastCouncilGrounding: groundings,
        lastDecisionCard: decisionCard,
        chatLogs: {
          ...prev.chatLogs,
          ...Object.fromEntries(selectedCouncilKeys.map(key => [key, [...(prev.chatLogs[key] || []), { role: 'user', content: userInput }, { role: 'assistant', content: responses[key] }]]))
        }
      }));
    } catch (err: any) { alert(err.message); } finally { setIsLoading(false); setUserInput(''); }
  };

  const sendMessage = async () => {
    if (!userInput.trim()) return;
    setIsLoading(true);
    const gemini = new GeminiService();
    const key = selectedAnalystKey;
    const analyst = ANALYSTS.find(a => a.key === key)!;
    const history = state.chatLogs[key] || [];
    try {
      const out = await gemini.generate(analyst.systemPrompt, userInput, history, state.settings);
      setState(prev => ({
        ...prev,
        chatLogs: { ...prev.chatLogs, [key]: [...history, { role: 'user', content: userInput }, { role: 'assistant', content: out.text }] }
      }));
    } catch (err: any) { alert(err.message); } finally { setIsLoading(false); setUserInput(''); }
  };

  if (isApiKeyMissing) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-950 p-6">
        <div className="w-full max-w-md bg-slate-900 border border-rose-500/30 p-8 rounded-3xl shadow-2xl text-center space-y-6">
          <div className="w-20 h-20 bg-rose-500/10 rounded-full flex items-center justify-center mx-auto ring-4 ring-rose-500/20">
            <ServerCrash className="w-10 h-10 text-rose-500" />
          </div>
          <h1 className="text-2xl font-bold text-slate-100 uppercase tracking-tight">Core Link Failure</h1>
          <p className="text-slate-400 text-sm">The Google API Key is missing. Neural synchronization cannot be established.</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-950 p-6">
        <form onSubmit={handleLogin} className="w-full max-w-md glass-card p-10 rounded-[2.5rem] shadow-2xl space-y-8">
          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-indigo-500/10 rounded-3xl flex items-center justify-center mx-auto mb-6 rotate-12">
              <Lock className="w-8 h-8 text-indigo-400" />
            </div>
            <h1 className="text-3xl font-extrabold text-white tracking-tight">Neural Uplink</h1>
            <p className="text-slate-500 text-sm">Enter access protocol to initialize system.</p>
          </div>
          <input 
            type="password" 
            value={passwordInput} 
            onChange={(e) => setPasswordInput(e.target.value)}
            placeholder="Protocol Key" 
            className="w-full bg-slate-950/50 border border-slate-800 rounded-2xl px-6 py-4 text-center tracking-[0.5em] text-indigo-400 focus:ring-2 focus:ring-indigo-500 outline-none transition-all placeholder:tracking-normal placeholder:text-slate-700"
            autoFocus
          />
          <button type="submit" className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-4 rounded-2xl transition-all uppercase tracking-widest text-xs shadow-lg shadow-indigo-600/20">
            Initialize Link
          </button>
        </form>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden bg-slate-950 text-slate-100 font-sans selection:bg-indigo-500/30">
      {/* Dynamic Sidebar / Drawer */}
      <div 
        className={`fixed inset-0 z-50 bg-black/80 backdrop-blur-md lg:hidden transition-opacity duration-300 ${isSidebarOpen ? 'opacity-100 visible' : 'opacity-0 invisible'}`}
        onClick={() => setIsSidebarOpen(false)}
      />
      <aside className={`fixed lg:static inset-y-0 left-0 z-50 w-72 md:w-80 bg-slate-900 border-r border-slate-800 transition-transform duration-300 transform lg:translate-x-0 ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} overflow-hidden flex flex-col shadow-2xl`}>
        <div className="p-8 border-b border-slate-800 flex justify-between items-center">
          <div>
            <h1 className="text-xl font-black flex items-center gap-2 text-indigo-400 tracking-tight">
              <BrainCircuit className="w-6 h-6" /> Solo AI
            </h1>
            <div className="flex items-center gap-1.5 mt-1">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Protocol v2.1.0</span>
            </div>
          </div>
          <button onClick={() => setIsSidebarOpen(false)} className="lg:hidden p-2 text-slate-500 hover:text-white transition-colors">
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-8 no-scrollbar">
          {([1, 2, 3, 4] as TierLevel[]).map(tier => (
            <section key={tier} className="space-y-3">
              <h2 className={`text-[10px] font-black uppercase flex items-center gap-2 tracking-[0.2em] px-3 ${TIER_INFO[tier].color}`}>
                {React.createElement(TIER_INFO[tier].icon, { className: "w-3.5 h-3.5" })} {TIER_INFO[tier].name}
              </h2>
              <div className="space-y-1">
                {groupedAnalysts[tier].map(a => (
                  <button
                    key={a.key}
                    onClick={() => { setSelectedAnalystKey(a.key); setActiveTab('single'); setIsSidebarOpen(false); }}
                    className={`w-full text-left p-3.5 rounded-2xl text-sm transition-all flex items-center gap-4 group ${selectedAnalystKey === a.key && activeTab === 'single' ? 'bg-indigo-600 text-white shadow-xl shadow-indigo-600/10' : 'text-slate-400 hover:bg-slate-800/50'}`}
                  >
                    <span className={`text-xl transition-transform group-hover:scale-110 ${selectedAnalystKey === a.key && activeTab === 'single' ? '' : 'filter grayscale'}`}>{a.icon}</span>
                    <div className="flex-1 truncate font-semibold">{a.name}</div>
                  </button>
                ))}
              </div>
            </section>
          ))}
        </div>

        <div className="p-4 mt-auto border-t border-slate-800">
          <button className="w-full flex items-center gap-3 p-4 text-xs font-bold text-slate-500 hover:text-indigo-400 hover:bg-indigo-500/5 rounded-2xl transition-all">
            <SettingsIcon className="w-4 h-4" /> Global Settings
          </button>
        </div>
      </aside>

      <main className="flex-1 flex flex-col bg-[#020617] relative">
        {/* Responsive Navbar */}
        <nav className="h-16 md:h-20 border-b border-slate-800 flex items-center px-4 md:px-8 gap-4 bg-slate-900/50 backdrop-blur-xl sticky top-0 z-40">
          <button onClick={() => setIsSidebarOpen(true)} className="lg:hidden p-2 text-slate-400">
            <Menu className="w-6 h-6" />
          </button>
          
          <div className="flex-1 flex items-center justify-center lg:justify-start gap-1 md:gap-6 overflow-x-auto no-scrollbar scroll-smooth">
            {[
              { id: 'council', icon: Users, label: 'Council' },
              { id: 'single', icon: User, label: 'Expert' },
              { id: 'live', icon: Mic, label: 'War Room' },
              { id: 'lab', icon: ImageIcon, label: 'Media Lab' }
            ].map(t => (
              <button 
                key={t.id} 
                onClick={() => setActiveTab(t.id as any)} 
                className={`flex items-center gap-2 text-xs md:text-sm font-bold transition-all px-4 py-2.5 rounded-xl whitespace-nowrap ${activeTab === t.id ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20' : 'text-slate-500 hover:text-slate-300'}`}
              >
                <t.icon className="w-4 h-4" /> <span className="hidden sm:inline">{t.label}</span>
              </button>
            ))}
          </div>

          <div className="flex items-center gap-3">
            <div className={`hidden md:flex h-9 w-9 rounded-full items-center justify-center border border-slate-800 transition-colors ${state.settings.enableSearch ? 'bg-indigo-500/10 border-indigo-500/50 text-indigo-400' : 'text-slate-500'}`}>
              <Search className="w-4 h-4" />
            </div>
            {isLiveActive && <div className="flex items-center gap-2 text-rose-500 text-[10px] font-black uppercase tracking-tighter bg-rose-500/10 px-3 py-1.5 rounded-full border border-rose-500/20 animate-pulse">Live</div>}
          </div>
        </nav>

        <div className="flex-1 overflow-y-auto p-4 md:p-10 no-scrollbar pb-32">
          {activeTab === 'council' && (
            <div className="max-w-6xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="glass-card rounded-[2.5rem] p-6 md:p-10 shadow-2xl overflow-hidden relative">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-indigo-500 via-emerald-500 to-rose-500 opacity-50" />
                
                <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 mb-10">
                  <div className="space-y-1">
                    <h3 className="text-2xl font-black text-white flex items-center gap-3 tracking-tight">
                      <LayoutDashboard className="w-6 h-6 text-indigo-400" /> Neural Council
                    </h3>
                    <p className="text-slate-500 text-xs font-semibold">Deploy multi-agent consensus protocols.</p>
                  </div>
                  <div className="flex flex-wrap gap-2 w-full md:w-auto">
                    {MODES.map(m => (
                      <button 
                        key={m.name} 
                        onClick={() => setSelectedCouncilKeys(m.keys)} 
                        className={`flex-1 md:flex-none px-5 py-2 rounded-full text-[10px] font-bold uppercase border border-slate-700 hover:border-indigo-500 transition-all bg-gradient-to-r ${m.color} text-slate-300 hover:text-white`}
                      >
                        {m.name}
                      </button>
                    ))}
                  </div>
                </div>
                
                <div className="flex overflow-x-auto gap-4 pb-6 no-scrollbar touch-pan-x">
                  {ANALYSTS.map(a => (
                    <button 
                      key={a.key} 
                      onClick={() => toggleCouncilRole(a.key)} 
                      className={`flex-shrink-0 w-24 h-28 p-4 rounded-3xl border transition-all flex flex-col items-center justify-center gap-3 group relative ${selectedCouncilKeys.includes(a.key) ? 'bg-indigo-600 border-indigo-400 shadow-lg shadow-indigo-600/30' : 'bg-slate-950/50 border-slate-800 hover:border-slate-600'}`}
                    >
                      <div className={`text-2xl group-hover:scale-110 transition-transform ${selectedCouncilKeys.includes(a.key) ? '' : 'filter grayscale'}`}>{a.icon}</div>
                      <div className="text-[8px] font-black uppercase text-center tracking-tighter opacity-80">{a.name}</div>
                      {selectedCouncilKeys.includes(a.key) && <div className="absolute -top-1 -right-1 w-4 h-4 bg-white rounded-full flex items-center justify-center"><Zap className="w-2 h-2 text-indigo-600 fill-current" /></div>}
                    </button>
                  ))}
                </div>

                <div className="relative mt-4">
                  <textarea 
                    value={userInput} 
                    onChange={(e) => setUserInput(e.target.value)} 
                    placeholder="Broadcast baseline parameters to the Neural Council..." 
                    className="w-full bg-slate-950/80 border border-slate-800 rounded-[2rem] p-8 text-sm min-h-[160px] focus:ring-4 focus:ring-indigo-500/10 focus:border-indigo-500 outline-none transition-all resize-none text-slate-200 placeholder:text-slate-700 font-medium" 
                  />
                  <div className="absolute bottom-6 right-6 flex items-center gap-3">
                    <button 
                      disabled={isLoading || !userInput.trim()} 
                      onClick={runCouncil} 
                      className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-black py-4 px-8 rounded-2xl flex items-center gap-3 uppercase tracking-widest text-xs transition-all shadow-xl shadow-indigo-600/20 active:scale-95"
                    >
                      {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Sparkles className="w-5 h-5" />} Execute Synthesis
                    </button>
                  </div>
                </div>
              </div>

              {state.lastDecisionCard && (
                <div className="glass-card rounded-[2.5rem] overflow-hidden shadow-2xl animate-in zoom-in-95 duration-700">
                  <div className="px-8 py-6 bg-indigo-500/5 border-b border-white/5 flex flex-col sm:flex-row justify-between items-center gap-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-indigo-500/10 rounded-xl"><Command className="w-4 h-4 text-indigo-400" /></div>
                      <h4 className="font-bold text-indigo-100 uppercase tracking-[0.2em] text-xs">Primary Synthesis Directive</h4>
                    </div>
                    <button onClick={playBriefing} disabled={isBriefing} className="w-full sm:w-auto flex items-center justify-center gap-3 px-8 py-3 bg-indigo-600 hover:bg-indigo-500 rounded-2xl text-[10px] font-black uppercase transition-all shadow-lg shadow-indigo-600/10">
                      {isBriefing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Volume2 className="w-4 h-4" />} Audio Briefing
                    </button>
                  </div>
                  <div className="p-8 md:p-12 prose prose-invert prose-indigo max-w-none prose-sm leading-relaxed">
                    <ReactMarkdown>{state.lastDecisionCard}</ReactMarkdown>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pb-20">
                {selectedCouncilKeys.map(key => {
                  const a = ANALYSTS.find(x => x.key === key)!;
                  const resp = state.lastCouncilResponses[key];
                  if (!resp) return null;
                  return (
                    <div key={a.key} className="glass-card rounded-3xl overflow-hidden hover:border-indigo-500/50 transition-all group">
                      <div className={`h-1.5 w-full ${a.color} opacity-40 group-hover:opacity-100 transition-opacity`} />
                      <div className="p-6 border-b border-white/5 flex items-center gap-4">
                        <span className="text-3xl filter drop-shadow-lg">{a.icon}</span>
                        <div>
                          <div className="text-[10px] font-black text-slate-300 uppercase tracking-widest">{a.name}</div>
                          <div className="text-[8px] text-slate-500 font-bold uppercase mt-0.5 tracking-tighter">Tier {a.tier} Executive</div>
                        </div>
                      </div>
                      <div className="p-6 text-[13px] text-slate-400 max-h-[400px] overflow-y-auto no-scrollbar leading-relaxed">
                        <ReactMarkdown className="prose prose-invert prose-xs">{resp}</ReactMarkdown>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {activeTab === 'single' && (
            <div className="max-w-4xl mx-auto flex flex-col h-full space-y-6">
              <div className="glass-card p-6 md:p-8 rounded-[2rem] flex items-center gap-6 shadow-2xl">
                <div className="text-5xl md:text-6xl p-4 bg-slate-950/50 rounded-3xl border border-white/5 rotate-3">{ANALYSTS.find(a => a.key === selectedAnalystKey)?.icon}</div>
                <div className="flex-1 overflow-hidden">
                  <h2 className="text-lg md:text-2xl font-black text-white truncate tracking-tight">{ANALYSTS.find(a => a.key === selectedAnalystKey)?.title}</h2>
                  <div className="flex items-center gap-3 mt-2">
                    <span className="px-3 py-1 bg-indigo-500/10 text-indigo-400 text-[10px] font-black uppercase rounded-lg border border-indigo-500/20">Active Session</span>
                    <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Focus: {ANALYSTS.find(a => a.key === selectedAnalystKey)?.focus}</span>
                  </div>
                </div>
              </div>
              
              <div className="flex-1 space-y-6 overflow-y-auto no-scrollbar py-4">
                {(state.chatLogs[selectedAnalystKey] || []).length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-slate-700 space-y-4 opacity-50">
                    <Command className="w-16 h-16 animate-pulse" />
                    <p className="text-[10px] font-black uppercase tracking-[0.3em]">Initialize Uplink</p>
                  </div>
                ) : (
                  (state.chatLogs[selectedAnalystKey] || []).map((msg, i) => (
                    <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in slide-in-from-bottom-2`}>
                      <div className={`max-w-[85%] p-5 rounded-3xl ${msg.role === 'user' ? 'bg-indigo-600 text-white shadow-xl shadow-indigo-600/10 rounded-tr-none' : 'glass-card text-slate-200 rounded-tl-none'}`}>
                        <ReactMarkdown className="prose prose-invert prose-sm leading-relaxed">{msg.content}</ReactMarkdown>
                      </div>
                    </div>
                  ))
                )}
                <div ref={chatEndRef} />
              </div>
            </div>
          )}

          {activeTab === 'live' && (
            <div className="h-full flex flex-col items-center justify-center max-w-2xl mx-auto space-y-16 py-12">
              <div className={`relative w-64 h-64 md:w-80 md:h-80 flex items-center justify-center rounded-[3rem] transition-all duration-1000 ${isLiveActive ? 'bg-indigo-600/10 shadow-[0_0_100px_rgba(79,70,229,0.3)] rotate-45 scale-110' : 'bg-slate-900 border-2 border-slate-800'}`}>
                {isLiveActive && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="absolute inset-0 rounded-[3rem] border border-indigo-500/20 animate-ping" />
                    <div className="absolute inset-8 rounded-[3rem] border-2 border-indigo-400/40 animate-pulse" />
                  </div>
                )}
                <BrainCircuit className={`w-32 h-32 md:w-40 md:h-40 transition-all duration-700 ${isLiveActive ? 'text-indigo-400 -rotate-45' : 'text-slate-700'}`} />
              </div>
              
              <div className="text-center space-y-6">
                <h2 className="text-3xl md:text-5xl font-black tracking-tighter text-white">{isLiveActive ? 'Uplink Established' : 'Voice War Room'}</h2>
                <p className="text-slate-500 text-sm md:text-base font-medium max-w-md mx-auto leading-relaxed">Encrypted tactical voice channel with the Neural Council Chair. Ultra-low latency reasoning.</p>
              </div>

              <button 
                onClick={startLiveWarRoom}
                className={`flex items-center gap-6 px-12 py-6 rounded-full font-black uppercase tracking-[0.2em] transition-all text-xs shadow-2xl ${isLiveActive ? 'bg-rose-600 hover:bg-rose-500 text-white shadow-rose-900/20' : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-900/20'}`}
              >
                {isLiveActive ? <><Square className="w-5 h-5 fill-current" /> Terminate Link</> : <><Mic className="w-5 h-5" /> Initialize Uplink</>}
              </button>
            </div>
          )}

          {activeTab === 'lab' && (
            <div className="max-w-4xl mx-auto space-y-10 py-6">
              <div className="glass-card rounded-[2.5rem] p-8 md:p-12 shadow-2xl space-y-10">
                <h3 className="text-2xl font-black flex items-center gap-4 text-white uppercase tracking-tight">
                  <ImageIcon className="w-8 h-8 text-indigo-400" /> Media Lab
                </h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                  <div onClick={() => document.getElementById('img-up')?.click()} className="aspect-square rounded-[3rem] border-4 border-dashed border-slate-800 flex flex-col items-center justify-center bg-slate-950/50 hover:bg-slate-900/80 transition-all cursor-pointer group overflow-hidden relative">
                    {state.currentImage ? <img src={state.currentImage} className="w-full h-full object-cover transition-transform group-hover:scale-110" /> : <div className="text-center p-8 space-y-4 text-slate-700 group-hover:text-slate-400 transition-colors"><Upload className="w-16 h-16 mx-auto mb-2 opacity-20" /><span className="text-xs font-black uppercase tracking-widest block">Input Source</span></div>}
                    <input type="file" id="img-up" hidden accept="image/*" onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (!f) return;
                      const r = new FileReader();
                      r.onload = (ev) => setState(p => ({ ...p, currentImage: ev.target?.result as string, editedImage: null }));
                      r.readAsDataURL(f);
                    }} />
                  </div>
                  <div className="aspect-square rounded-[3rem] border-2 border-slate-800 flex flex-col items-center justify-center bg-slate-950 overflow-hidden relative shadow-inner">
                    {state.editedImage ? <img src={state.editedImage} className="w-full h-full object-cover" /> : <div className="text-slate-900 text-[6rem] font-black opacity-10 select-none -rotate-12">RECON</div>}
                    {isLoading && <div className="absolute inset-0 bg-slate-950/90 flex flex-col items-center justify-center backdrop-blur-md animate-in fade-in"><Loader2 className="w-12 h-12 animate-spin text-indigo-500 mb-6" /><span className="text-[10px] font-black uppercase tracking-[0.3em] text-indigo-400">Reconstructing...</span></div>}
                  </div>
                </div>

                <div className="space-y-6">
                  <textarea 
                    value={userInput} 
                    onChange={(e) => setUserInput(e.target.value)} 
                    placeholder="Describe the desired visual transformation..." 
                    className="w-full bg-slate-950/50 border border-slate-800 rounded-3xl p-8 text-sm min-h-[120px] focus:ring-4 focus:ring-indigo-500/10 outline-none transition-all resize-none text-slate-200 placeholder:text-slate-800 font-medium" 
                  />
                  <button 
                    disabled={isLoading || !userInput.trim() || !state.currentImage} 
                    onClick={async () => {
                      setIsLoading(true);
                      const g = new GeminiService();
                      try {
                        const e = await g.editImage(userInput, state.currentImage!);
                        setState(p => ({ ...p, editedImage: e }));
                      } catch (err: any) { alert(err.message); } finally { setIsLoading(false); setUserInput(''); }
                    }} 
                    className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-black py-6 rounded-2xl uppercase tracking-[0.2em] text-xs transition-all shadow-xl shadow-indigo-600/20 active:scale-95"
                  >
                    Execute Media Edit
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Neural Input Bar - Dynamic Placement */}
        {activeTab !== 'live' && activeTab !== 'lab' && (
          <div className="fixed bottom-0 left-0 right-0 lg:left-80 p-4 md:p-10 bg-gradient-to-t from-slate-950 via-slate-950 to-transparent z-40">
            <div className="max-w-4xl mx-auto flex gap-4 p-3 bg-slate-900/90 backdrop-blur-2xl border border-white/10 rounded-[2rem] shadow-2xl items-center neural-pulse ring-1 ring-white/10">
              <div className="flex-1 px-4">
                <textarea 
                  value={userInput} 
                  onChange={(e) => setUserInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); activeTab === 'council' ? runCouncil() : sendMessage(); } }}
                  placeholder={`Establish Command (${activeTab === 'council' ? 'Council' : 'Expert'})...`}
                  className="w-full bg-transparent border-none outline-none text-sm py-3 max-h-32 resize-none no-scrollbar leading-relaxed font-medium placeholder:text-slate-600" 
                  rows={1}
                />
              </div>
              <button 
                onClick={activeTab === 'council' ? runCouncil : sendMessage} 
                disabled={isLoading || !userInput.trim()} 
                className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 p-4 rounded-2xl transition-all shadow-xl shadow-indigo-600/40 active:scale-90"
              >
                {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
              </button>
            </div>
          </div>
        )}
        
        {/* Mobile Safe Area Spacer */}
        <div className="h-[var(--safe-area-inset-bottom)] bg-slate-950" />
      </main>
    </div>
  );
};

export default App;
