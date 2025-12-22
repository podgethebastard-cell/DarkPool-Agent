
import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { 
  Settings as SettingsIcon, 
  Users, 
  User, 
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
  Square, 
  Menu, 
  X, 
  Lock, 
  ServerCrash,
  Sparkles,
  Command,
  Settings,
  ChevronRight,
  History,
  // Added Volume2 import
  Volume2
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

  const encode = (bytes: Uint8Array) => {
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
  };

  const decode = (base64: string) => {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
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

  const toggleCouncilRole = (key: string) => {
    setSelectedCouncilKeys(prev => 
      prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]
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
          const base64 = encode(new Uint8Array(int16.buffer));
          sessionPromise.then(s => s.sendRealtimeInput({ media: { data: base64, mimeType: 'audio/pcm;rate=16000' } }));
        };
        source.connect(scriptProcessor);
        scriptProcessor.connect(inputCtx.destination);
      },
      onmessage: async (msg) => {
        const audioData = msg.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
        if (audioData && audioContextRef.current) {
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
    }, "Facilitate a high-stakes war room session. Be fast, direct, and authoritative.");
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
      <div className="flex items-center justify-center min-h-screen bg-slate-950 p-6 text-center">
        <div className="glass-card p-10 rounded-[2rem] border-rose-500/20 max-w-sm space-y-6">
          <ServerCrash className="w-16 h-16 text-rose-500 mx-auto animate-pulse" />
          <h1 className="text-xl font-bold uppercase tracking-widest">Neural Link Offline</h1>
          <p className="text-slate-500 text-sm">Google API Key missing. Check environment secrets.</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-950 p-6">
        <form onSubmit={handleLogin} className="w-full max-w-md glass-card p-10 rounded-[2.5rem] shadow-2xl space-y-8 animate-in zoom-in-95 duration-500">
          <div className="text-center space-y-4">
            <div className="w-20 h-20 bg-indigo-500/10 rounded-3xl flex items-center justify-center mx-auto mb-6 rotate-6 border border-indigo-500/20">
              <Lock className="w-10 h-10 text-indigo-400" />
            </div>
            <h1 className="text-3xl font-black text-white tracking-tight uppercase">Neural Guard</h1>
            <p className="text-slate-500 text-sm font-medium">Identity verification required for uplink.</p>
          </div>
          <input 
            type="password" 
            value={passwordInput} 
            onChange={(e) => setPasswordInput(e.target.value)}
            placeholder="ACCESS PROTOCOL" 
            className="w-full bg-slate-950/50 border border-slate-800 rounded-2xl px-6 py-5 text-center tracking-[0.6em] text-indigo-400 focus:ring-2 focus:ring-indigo-500 outline-none transition-all placeholder:tracking-normal placeholder:text-slate-800 font-mono"
            autoFocus
          />
          <button type="submit" className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-black py-5 rounded-2xl transition-all uppercase tracking-[0.2em] text-xs shadow-xl shadow-indigo-600/20 active:scale-95">
            Initialize Link
          </button>
        </form>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden bg-[#020617] text-slate-100 font-sans selection:bg-indigo-500/30">
      {/* Desktop Sidebar */}
      <aside className={`fixed lg:static inset-y-0 left-0 z-50 w-80 bg-slate-900 border-r border-slate-800/60 transition-transform duration-300 transform lg:translate-x-0 ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} flex flex-col shadow-2xl`}>
        <div className="p-8 border-b border-slate-800/40 flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-black flex items-center gap-2 text-indigo-400 tracking-tighter">
              <BrainCircuit className="w-8 h-8" /> SOLO AI
            </h1>
            <div className="flex items-center gap-1.5 mt-1">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.8)]" />
              <span className="text-[10px] text-slate-500 uppercase tracking-widest font-black">Online • Protocol 2.1</span>
            </div>
          </div>
          <button onClick={() => setIsSidebarOpen(false)} className="lg:hidden p-2 text-slate-500 hover:text-white transition-colors">
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-8 no-scrollbar scroll-smooth">
          {([1, 2, 3, 4] as TierLevel[]).map(tier => (
            <section key={tier} className="space-y-4">
              <h2 className={`text-[10px] font-black uppercase flex items-center gap-2 tracking-[0.25em] px-4 py-1.5 rounded-lg ${TIER_INFO[tier].bg} ${TIER_INFO[tier].color} border border-white/5`}>
                {React.createElement(TIER_INFO[tier].icon, { className: "w-3.5 h-3.5" })} {TIER_INFO[tier].name}
              </h2>
              <div className="space-y-1">
                {groupedAnalysts[tier].map(a => (
                  <button
                    key={a.key}
                    onClick={() => { setSelectedAnalystKey(a.key); setActiveTab('single'); setIsSidebarOpen(false); }}
                    className={`w-full text-left p-3.5 rounded-2xl text-sm transition-all flex items-center gap-4 group ${selectedAnalystKey === a.key && activeTab === 'single' ? 'bg-indigo-600 text-white shadow-xl shadow-indigo-600/20' : 'text-slate-400 hover:bg-slate-800/40'}`}
                  >
                    <span className={`text-xl transition-transform group-hover:scale-125 duration-300 ${selectedAnalystKey === a.key && activeTab === 'single' ? 'scale-110' : 'opacity-40 filter grayscale'}`}>{a.icon}</span>
                    <div className="flex-1 truncate font-bold tracking-tight">{a.name}</div>
                    {selectedAnalystKey === a.key && activeTab === 'single' && <ChevronRight className="w-4 h-4 opacity-50" />}
                  </button>
                ))}
              </div>
            </section>
          ))}
        </div>
      </aside>

      <main className="flex-1 flex flex-col bg-[#020617] relative">
        {/* Navbar */}
        <nav className="h-16 md:h-20 border-b border-slate-800/40 flex items-center px-4 md:px-8 gap-4 bg-slate-900/60 backdrop-blur-2xl sticky top-0 z-40">
          <button onClick={() => setIsSidebarOpen(true)} className="lg:hidden p-3 bg-slate-800/40 rounded-xl text-slate-400">
            <Menu className="w-6 h-6" />
          </button>
          
          <div className="flex-1 flex items-center justify-center lg:justify-start gap-2 md:gap-6 overflow-x-auto no-scrollbar scroll-smooth py-2">
            {[
              { id: 'council', icon: Users, label: 'Council' },
              { id: 'single', icon: User, label: 'Expert' },
              { id: 'live', icon: Mic, label: 'Live' },
              { id: 'lab', icon: ImageIcon, label: 'Media' }
            ].map(t => (
              <button 
                key={t.id} 
                onClick={() => setActiveTab(t.id as any)} 
                className={`flex items-center gap-2 text-xs md:text-sm font-black transition-all px-5 py-2.5 rounded-2xl whitespace-nowrap uppercase tracking-widest ${activeTab === t.id ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30' : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/40'}`}
              >
                <t.icon className="w-4 h-4" /> <span className="hidden sm:inline">{t.label}</span>
              </button>
            ))}
          </div>

          <div className="flex items-center gap-3">
            <button 
              onClick={() => setState(prev => ({ ...prev, settings: { ...prev.settings, enableSearch: !prev.settings.enableSearch }}))}
              className={`w-10 h-10 rounded-2xl flex items-center justify-center border transition-all ${state.settings.enableSearch ? 'bg-indigo-600/10 border-indigo-500/50 text-indigo-400 shadow-[0_0_12px_rgba(79,70,229,0.2)]' : 'bg-slate-800/40 border-slate-700 text-slate-600'}`}
            >
              <Search className="w-4 h-4" />
            </button>
            {isLiveActive && (
              <div className="flex items-center gap-2 text-rose-500 text-[9px] font-black uppercase tracking-widest bg-rose-500/10 px-4 py-2 rounded-full border border-rose-500/20 shadow-[0_0_15px_rgba(244,63,94,0.3)] animate-pulse">
                Recording
              </div>
            )}
          </div>
        </nav>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-4 md:p-10 no-scrollbar pb-40 scroll-smooth">
          {activeTab === 'council' && (
            <div className="max-w-6xl mx-auto space-y-10 animate-in fade-in slide-in-from-bottom-4 duration-700">
              <div className="glass-card rounded-[3rem] p-6 md:p-12 shadow-[0_20px_50px_rgba(0,0,0,0.5)] border-white/5 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-indigo-500 via-emerald-500 to-rose-500 opacity-60" />
                
                <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-8 mb-12">
                  <div className="space-y-2">
                    <h3 className="text-3xl font-black text-white flex items-center gap-4 tracking-tighter">
                      <LayoutDashboard className="w-8 h-8 text-indigo-400" /> NEURAL COUNCIL
                    </h3>
                    <p className="text-slate-500 text-sm font-bold tracking-wide uppercase opacity-70">Decentralized Execution Protocol</p>
                  </div>
                  <div className="flex flex-wrap gap-2 w-full md:w-auto">
                    {MODES.map(m => (
                      <button 
                        key={m.name} 
                        onClick={() => setSelectedCouncilKeys(m.keys)} 
                        className={`flex-1 md:flex-none px-6 py-2.5 rounded-full text-[10px] font-black uppercase border border-white/10 transition-all bg-gradient-to-r ${m.color} text-slate-300 hover:text-white hover:scale-105 active:scale-95`}
                      >
                        {m.name}
                      </button>
                    ))}
                  </div>
                </div>
                
                <div className="flex overflow-x-auto gap-4 pb-8 no-scrollbar touch-pan-x px-2">
                  {ANALYSTS.map(a => (
                    <button 
                      key={a.key} 
                      onClick={() => toggleCouncilRole(a.key)} 
                      className={`flex-shrink-0 w-28 h-32 p-5 rounded-[2rem] border transition-all flex flex-col items-center justify-center gap-4 group relative ${selectedCouncilKeys.includes(a.key) ? 'bg-indigo-600 border-indigo-300 shadow-2xl shadow-indigo-600/40 ring-4 ring-indigo-500/20' : 'bg-slate-950/60 border-slate-800/60 hover:border-slate-500'}`}
                    >
                      <div className={`text-3xl transition-transform group-hover:scale-125 duration-300 ${selectedCouncilKeys.includes(a.key) ? 'scale-110 drop-shadow-[0_0_10px_rgba(255,255,255,0.5)]' : 'opacity-40 filter grayscale'}`}>{a.icon}</div>
                      <div className="text-[9px] font-black uppercase text-center tracking-tighter leading-none opacity-80">{a.name}</div>
                      {selectedCouncilKeys.includes(a.key) && <div className="absolute -top-1 -right-1 w-6 h-6 bg-white rounded-full flex items-center justify-center shadow-lg"><Zap className="w-3 h-3 text-indigo-600 fill-current" /></div>}
                    </button>
                  ))}
                </div>

                <div className="relative mt-6">
                  <textarea 
                    value={userInput} 
                    onChange={(e) => setUserInput(e.target.value)} 
                    placeholder="Broadcast baseline parameters to the Neural Council..." 
                    className="w-full bg-slate-950/80 border border-slate-800/80 rounded-[2.5rem] p-8 text-sm min-h-[180px] focus:ring-4 focus:ring-indigo-500/20 focus:border-indigo-500 outline-none transition-all resize-none text-slate-200 placeholder:text-slate-800 font-bold leading-relaxed shadow-inner" 
                  />
                  <div className="absolute bottom-8 right-8">
                    <button 
                      disabled={isLoading || !userInput.trim()} 
                      onClick={runCouncil} 
                      className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white font-black py-5 px-10 rounded-3xl flex items-center gap-4 uppercase tracking-[0.2em] text-xs transition-all shadow-2xl shadow-indigo-600/30 active:scale-95 group"
                    >
                      {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Sparkles className="w-5 h-5 group-hover:rotate-12 transition-transform" />} Execute Synthesis
                    </button>
                  </div>
                </div>
              </div>

              {state.lastDecisionCard && (
                <div className="glass-card rounded-[3.5rem] overflow-hidden shadow-2xl animate-in zoom-in-95 duration-1000 border-indigo-500/20">
                  <div className="px-10 py-8 bg-indigo-500/5 border-b border-white/5 flex flex-col sm:flex-row justify-between items-center gap-6">
                    <div className="flex items-center gap-4">
                      <div className="p-3 bg-indigo-500/10 rounded-2xl border border-indigo-500/20"><Command className="w-5 h-5 text-indigo-400" /></div>
                      <div>
                        <h4 className="font-black text-indigo-100 uppercase tracking-[0.3em] text-[10px]">Neural Response Package</h4>
                        <p className="text-[9px] text-slate-500 font-black uppercase mt-1">Status: Verified Synthesis</p>
                      </div>
                    </div>
                    {/* Fixed Volume2 usage */}
                    <button onClick={playBriefing} disabled={isBriefing} className="w-full sm:w-auto flex items-center justify-center gap-4 px-10 py-4 bg-indigo-600 hover:bg-indigo-500 rounded-[1.5rem] text-[10px] font-black uppercase tracking-widest transition-all shadow-xl shadow-indigo-600/20 active:scale-95">
                      {isBriefing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Volume2 className="w-4 h-4" />} Audio Briefing
                    </button>
                  </div>
                  <div className="p-10 md:p-16 prose prose-invert prose-indigo max-w-none prose-sm leading-relaxed selection:bg-indigo-500/40">
                    <ReactMarkdown>{state.lastDecisionCard}</ReactMarkdown>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 pb-12">
                {selectedCouncilKeys.map(key => {
                  const a = ANALYSTS.find(x => x.key === key)!;
                  const resp = state.lastCouncilResponses[key];
                  if (!resp) return null;
                  return (
                    <div key={a.key} className="glass-card rounded-[2.5rem] overflow-hidden hover:border-indigo-500/40 transition-all group shadow-xl">
                      <div className={`h-2 w-full ${a.color} opacity-30 group-hover:opacity-100 transition-opacity`} />
                      <div className="p-8 border-b border-white/5 flex items-center gap-5">
                        <span className="text-4xl transition-transform group-hover:scale-110 duration-500">{a.icon}</span>
                        <div>
                          <div className="text-xs font-black text-slate-200 uppercase tracking-widest">{a.name}</div>
                          <div className="text-[9px] text-slate-500 font-black uppercase mt-1 tracking-tighter opacity-60">Tier {a.tier} Analyst</div>
                        </div>
                      </div>
                      {/* Fixed ReactMarkdown className usage by wrapping it */}
                      <div className="p-8 text-[13px] text-slate-400 max-h-[450px] overflow-y-auto no-scrollbar leading-relaxed">
                        <div className="prose prose-invert prose-xs">
                          <ReactMarkdown>{resp}</ReactMarkdown>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {activeTab === 'single' && (
            <div className="max-w-4xl mx-auto flex flex-col h-full space-y-8 animate-in fade-in duration-500">
              <div className="glass-card p-8 md:p-10 rounded-[3rem] flex items-center gap-8 shadow-2xl border-white/5">
                <div className="text-6xl md:text-7xl p-6 bg-slate-950/60 rounded-[2rem] border border-white/10 rotate-2 shadow-inner transition-transform hover:rotate-0 duration-500">{ANALYSTS.find(a => a.key === selectedAnalystKey)?.icon}</div>
                <div className="flex-1 overflow-hidden">
                  <h2 className="text-xl md:text-3xl font-black text-white truncate tracking-tighter uppercase">{ANALYSTS.find(a => a.key === selectedAnalystKey)?.title}</h2>
                  <div className="flex flex-wrap items-center gap-3 mt-4">
                    <span className="px-4 py-1.5 bg-indigo-500/10 text-indigo-400 text-[10px] font-black uppercase rounded-full border border-indigo-500/20 shadow-[0_0_10px_rgba(79,70,229,0.2)]">Neural Link: Active</span>
                    <span className="text-[10px] text-slate-500 font-black uppercase tracking-widest border border-white/5 px-4 py-1.5 rounded-full">Focus: {ANALYSTS.find(a => a.key === selectedAnalystKey)?.focus}</span>
                  </div>
                </div>
              </div>
              
              <div className="flex-1 space-y-8 overflow-y-auto no-scrollbar py-6">
                {(state.chatLogs[selectedAnalystKey] || []).length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-slate-800 space-y-6 opacity-30">
                    <div className="p-8 bg-slate-900/40 rounded-[3rem] border border-white/5 animate-pulse">
                      <Command className="w-20 h-20" />
                    </div>
                    <p className="text-xs font-black uppercase tracking-[0.4em]">Initialize Communication Sequence</p>
                  </div>
                ) : (
                  (state.chatLogs[selectedAnalystKey] || []).map((msg, i) => (
                    <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in slide-in-from-bottom-3`}>
                      {/* Fixed ReactMarkdown className usage by moving it to the parent container */}
                      <div className={`max-w-[85%] p-6 rounded-[2rem] shadow-2xl ${msg.role === 'user' ? 'bg-indigo-600 text-white shadow-indigo-600/20 rounded-tr-none' : 'glass-card text-slate-200 rounded-tl-none border-white/10'} prose prose-invert prose-sm leading-relaxed font-medium`}>
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>
                    </div>
                  ))
                )}
                <div ref={chatEndRef} />
              </div>
            </div>
          )}

          {activeTab === 'live' && (
            <div className="h-full flex flex-col items-center justify-center max-w-2xl mx-auto space-y-20 py-12 animate-in zoom-in-95 duration-700">
              <div className={`relative w-72 h-72 md:w-96 md:h-96 flex items-center justify-center rounded-[4rem] transition-all duration-1000 ${isLiveActive ? 'bg-indigo-600/10 shadow-[0_0_120px_rgba(79,70,229,0.3)] rotate-45 scale-110 border-indigo-500/40' : 'bg-slate-900/40 border-2 border-slate-800'}`}>
                {isLiveActive && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="absolute inset-0 rounded-[4rem] border border-indigo-500/30 animate-ping duration-[2000ms]" />
                    <div className="absolute inset-10 rounded-[3.5rem] border-2 border-indigo-400/20 animate-pulse duration-[1500ms]" />
                    <div className="absolute inset-20 rounded-[3rem] border border-indigo-400/10 animate-pulse duration-[3000ms]" />
                  </div>
                )}
                <BrainCircuit className={`w-36 h-36 md:w-48 md:h-48 transition-all duration-1000 ${isLiveActive ? 'text-indigo-400 -rotate-45 drop-shadow-[0_0_20px_rgba(129,140,248,0.5)]' : 'text-slate-800'}`} />
              </div>
              
              <div className="text-center space-y-8">
                <h2 className="text-4xl md:text-6xl font-black tracking-tighter text-white uppercase">{isLiveActive ? 'Uplink Established' : 'Voice War Room'}</h2>
                <div className="flex flex-col gap-4">
                  <p className="text-slate-500 text-sm md:text-lg font-bold max-w-md mx-auto leading-relaxed uppercase tracking-wide opacity-80">Encrypted tactical voice channel enabled. Multi-agent core synced.</p>
                  <div className="flex items-center justify-center gap-4 text-[10px] font-black uppercase text-indigo-400/60 tracking-widest">
                    <span>Low Latency</span>
                    <div className="w-1 h-1 bg-slate-800 rounded-full" />
                    <span>Neural Audio Core</span>
                    <div className="w-1 h-1 bg-slate-800 rounded-full" />
                    <span>Secure Endpoints</span>
                  </div>
                </div>
              </div>

              <button 
                onClick={startLiveWarRoom}
                className={`flex items-center gap-8 px-16 py-8 rounded-[2rem] font-black uppercase tracking-[0.3em] transition-all text-sm shadow-[0_20px_40px_rgba(0,0,0,0.4)] active:scale-95 ${isLiveActive ? 'bg-rose-600 hover:bg-rose-500 text-white shadow-rose-900/20 ring-4 ring-rose-500/10' : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-900/20 ring-4 ring-indigo-500/10'}`}
              >
                {isLiveActive ? <><Square className="w-6 h-6 fill-current" /> Terminate Link</> : <><Mic className="w-6 h-6" /> Initialize Uplink</>}
              </button>
            </div>
          )}

          {activeTab === 'lab' && (
            <div className="max-w-5xl mx-auto space-y-12 py-6 animate-in fade-in duration-700">
              <div className="glass-card rounded-[3.5rem] p-10 md:p-16 shadow-2xl space-y-12 border-white/5">
                <div className="flex items-center justify-between">
                  <h3 className="text-3xl font-black flex items-center gap-6 text-white uppercase tracking-tighter">
                    <ImageIcon className="w-10 h-10 text-indigo-400" /> Neural Lab
                  </h3>
                  <div className="text-[10px] font-black text-slate-600 uppercase tracking-[0.3em] bg-slate-950 px-5 py-2 rounded-full border border-white/5">Image Reconstruction Core</div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
                  <div onClick={() => document.getElementById('img-up-lab')?.click()} className="aspect-square rounded-[3.5rem] border-4 border-dashed border-slate-800/60 flex flex-col items-center justify-center bg-slate-950/40 hover:bg-slate-900/80 transition-all cursor-pointer group overflow-hidden relative shadow-inner">
                    {state.currentImage ? <img src={state.currentImage} className="w-full h-full object-cover transition-transform group-hover:scale-110 duration-700" /> : <div className="text-center p-10 space-y-6 text-slate-700 group-hover:text-slate-400 transition-colors"><Upload className="w-20 h-20 mx-auto mb-4 opacity-10" /><span className="text-xs font-black uppercase tracking-[0.3em] block">Injection Point</span></div>}
                    <input type="file" id="img-up-lab" hidden accept="image/*" onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (!f) return;
                      const r = new FileReader();
                      r.onload = (ev) => setState(p => ({ ...p, currentImage: ev.target?.result as string, editedImage: null }));
                      r.readAsDataURL(f);
                    }} />
                  </div>
                  <div className="aspect-square rounded-[3.5rem] border-2 border-slate-800/40 flex flex-col items-center justify-center bg-slate-950 overflow-hidden relative shadow-2xl">
                    {state.editedImage ? <img src={state.editedImage} className="w-full h-full object-cover animate-in fade-in duration-1000" /> : <div className="text-slate-900/10 text-[8rem] font-black select-none -rotate-12 uppercase tracking-tighter">Recon</div>}
                    {isLoading && <div className="absolute inset-0 bg-slate-950/90 flex flex-col items-center justify-center backdrop-blur-xl animate-in fade-in duration-300"><Loader2 className="w-16 h-16 animate-spin text-indigo-500 mb-8" /><span className="text-xs font-black uppercase tracking-[0.5em] text-indigo-400 animate-pulse">Reconstructing Neural Map...</span></div>}
                  </div>
                </div>

                <div className="space-y-8 relative">
                  <textarea 
                    value={userInput} 
                    onChange={(e) => setUserInput(e.target.value)} 
                    placeholder="Provide vision directives for the neural engine..." 
                    className="w-full bg-slate-950/60 border border-slate-800/80 rounded-[2.5rem] p-10 text-sm min-h-[140px] focus:ring-4 focus:ring-indigo-500/10 outline-none transition-all resize-none text-slate-200 placeholder:text-slate-800 font-bold leading-relaxed shadow-inner" 
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
                    className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white font-black py-7 rounded-[1.5rem] uppercase tracking-[0.4em] text-xs transition-all shadow-2xl shadow-indigo-600/30 active:scale-95"
                  >
                    Deploy Image Matrix
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Floating Action Bar (Bottom Mobile) */}
        {activeTab !== 'live' && activeTab !== 'lab' && (
          <div className="fixed bottom-0 left-0 right-0 lg:left-80 p-4 md:p-12 bg-gradient-to-t from-slate-950 via-slate-950/80 to-transparent z-40">
            <div className="max-w-4xl mx-auto flex gap-4 p-4 bg-slate-900/95 backdrop-blur-3xl border border-white/10 rounded-[2.5rem] shadow-[0_25px_60px_-15px_rgba(0,0,0,0.8)] items-center neural-pulse ring-1 ring-white/10">
              <div className="flex-1 px-4">
                <textarea 
                  value={userInput} 
                  onChange={(e) => setUserInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); activeTab === 'council' ? runCouncil() : sendMessage(); } }}
                  placeholder={`Consult ${activeTab === 'council' ? 'Council' : 'Expert'} Analyst...`}
                  className="w-full bg-transparent border-none outline-none text-sm py-3 max-h-32 resize-none no-scrollbar leading-relaxed font-bold placeholder:text-slate-700" 
                  rows={1}
                />
              </div>
              <button 
                onClick={activeTab === 'council' ? runCouncil : sendMessage} 
                disabled={isLoading || !userInput.trim()} 
                className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-30 p-4 rounded-2xl transition-all shadow-2xl shadow-indigo-600/40 active:scale-90"
              >
                {isLoading ? <Loader2 className="w-6 h-6 animate-spin" /> : <Send className="w-6 h-6" />}
              </button>
            </div>
          </div>
        )}

        {/* Mobile App Bar */}
        <div className="lg:hidden fixed bottom-0 left-0 right-0 h-[72px] bg-slate-900/80 backdrop-blur-3xl border-t border-white/5 px-6 flex items-center justify-between z-50 pb-[env(safe-area-inset-bottom)]">
          {[
            { id: 'council', icon: Users, label: 'Council' },
            { id: 'single', icon: User, label: 'Single' },
            { id: 'live', icon: Mic, label: 'Live' },
            { id: 'lab', icon: ImageIcon, label: 'Lab' }
          ].map(t => (
            <button 
              key={t.id} 
              onClick={() => setActiveTab(t.id as any)} 
              className={`flex flex-col items-center gap-1.5 transition-all ${activeTab === t.id ? 'text-indigo-400 scale-110' : 'text-slate-500'}`}
            >
              <t.icon className={`w-6 h-6 ${activeTab === t.id ? 'drop-shadow-[0_0_8px_rgba(129,140,248,0.5)]' : ''}`} />
              <span className="text-[9px] font-black uppercase tracking-widest">{t.label}</span>
            </button>
          ))}
          <button onClick={() => setIsSidebarOpen(true)} className="flex flex-col items-center gap-1.5 text-slate-500">
            <Menu className="w-6 h-6" />
            <span className="text-[9px] font-black uppercase tracking-widest">Menu</span>
          </button>
        </div>
        
        <div className="h-[env(safe-area-inset-bottom)] bg-slate-900" />
      </main>
    </div>
  );
};

export default App;
