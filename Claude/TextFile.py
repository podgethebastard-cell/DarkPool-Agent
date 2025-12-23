<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ultimate Text File Creator</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            flex-wrap: wrap;
            gap: 20px;
        }

        h1 {
            color: #333;
            font-size: 28px;
        }

        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            border-bottom: 2px solid #e0e0e0;
            overflow-x: auto;
        }

        .tab {
            padding: 12px 20px;
            background: none;
            border: none;
            border-bottom: 3px solid transparent;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            color: #666;
            transition: all 0.3s;
            white-space: nowrap;
        }

        .tab:hover {
            color: #667eea;
        }

        .tab.active {
            color: #667eea;
            border-bottom-color: #667eea;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .two-column {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 30px;
        }

        .input-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 500;
            font-size: 14px;
        }

        input[type="text"], select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }

        input[type="text"]:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }

        textarea {
            width: 100%;
            min-height: 400px;
            padding: 16px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            resize: vertical;
            transition: border-color 0.3s;
            line-height: 1.6;
        }

        textarea:focus {
            outline: none;
            border-color: #667eea;
        }

        textarea.monospace {
            font-family: 'Courier New', monospace;
        }

        .toolbar {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }

        .toolbar button {
            padding: 8px 16px;
            background: #f5f5f5;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }

        .toolbar button:hover {
            background: #e0e0e0;
        }

        .button-group {
            display: flex;
            gap: 12px;
            margin-top: 20px;
            flex-wrap: wrap;
        }

        button.primary {
            flex: 1;
            min-width: 150px;
            padding: 14px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        button.primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }

        button.secondary {
            flex: 1;
            min-width: 150px;
            padding: 14px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            background: #f5f5f5;
            color: #666;
        }

        button.secondary:hover {
            background: #e0e0e0;
        }

        .sidebar {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 12px;
        }

        .sidebar h3 {
            font-size: 16px;
            margin-bottom: 16px;
            color: #333;
        }

        .char-count {
            color: #666;
            font-size: 13px;
            margin-bottom: 20px;
            line-height: 1.6;
        }

        .saved-files {
            max-height: 300px;
            overflow-y: auto;
        }

        .file-item {
            background: white;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            transition: all 0.2s;
        }

        .file-item:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .file-item-name {
            font-size: 14px;
            color: #333;
            font-weight: 500;
        }

        .file-item-actions {
            display: flex;
            gap: 8px;
        }

        .file-item button {
            padding: 4px 8px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }

        .file-item button.delete {
            background: #f44336;
        }

        .success-message {
            background: #4caf50;
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            margin-top: 20px;
            display: none;
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .info-box {
            background: #f0f4ff;
            border-left: 4px solid #667eea;
            padding: 16px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 13px;
            color: #555;
        }

        .format-options {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }

        .format-btn {
            padding: 10px;
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            cursor: pointer;
            text-align: center;
            transition: all 0.2s;
            font-size: 13px;
        }

        .format-btn:hover {
            border-color: #667eea;
            background: #f0f4ff;
        }

        .format-btn.selected {
            border-color: #667eea;
            background: #667eea;
            color: white;
        }

        .drop-zone {
            border: 3px dashed #ccc;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s;
            margin-bottom: 20px;
            cursor: pointer;
        }

        .drop-zone.drag-over {
            border-color: #667eea;
            background: #f0f4ff;
        }

        .drop-zone p {
            color: #666;
            margin-bottom: 10px;
        }

        .template-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .template-card {
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            padding: 20px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .template-card:hover {
            border-color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .template-card h4 {
            color: #333;
            margin-bottom: 8px;
            font-size: 16px;
        }

        .template-card p {
            color: #666;
            font-size: 13px;
        }

        @media (max-width: 768px) {
            .two-column {
                grid-template-columns: 1fr;
            }
            
            .container {
                padding: 20px;
            }

            .button-group {
                flex-direction: column;
            }

            button.primary, button.secondary {
                width: 100%;
            }
        }

        .guide-section {
            background: #f9f9f9;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .guide-section h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 18px;
        }

        .guide-step {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }

        .guide-step strong {
            color: #667eea;
            display: block;
            margin-bottom: 8px;
        }

        .code-box {
            background: #2d2d2d;
            color: #f8f8f8;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            overflow-x: auto;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>📝 Ultimate Text File Creator</h1>
                <p style="color: #666; font-size: 14px; margin-top: 8px;">Create, format, and manage multiple text files with templates & batch export</p>
            </div>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="switchTab('editor')">✍️ Editor</button>
            <button class="tab" onclick="switchTab('templates')">📋 Templates</button>
            <button class="tab" onclick="switchTab('batch')">📚 Batch Create</button>
            <button class="tab" onclick="switchTab('import')">📂 Import Files</button>
            <button class="tab" onclick="switchTab('guide')">🚀 Netlify Guide</button>
        </div>

        <!-- Editor Tab -->
        <div id="editor-tab" class="tab-content active">
            <div class="two-column">
                <div>
                    <div class="input-group">
                        <label for="filename">Filename</label>
                        <input type="text" id="filename" placeholder="my-file" value="my-text-file">
                    </div>

                    <div class="input-group">
                        <label>File Format</label>
                        <div class="format-options">
                            <div class="format-btn selected" onclick="selectFormat('txt')">📄 TXT</div>
                            <div class="format-btn" onclick="selectFormat('md')">📝 MD</div>
                            <div class="format-btn" onclick="selectFormat('html')">🌐 HTML</div>
                            <div class="format-btn" onclick="selectFormat('csv')">📊 CSV</div>
                            <div class="format-btn" onclick="selectFormat('json')">🔧 JSON</div>
                            <div class="format-btn" onclick="selectFormat('xml')">📋 XML</div>
                        </div>
                    </div>

                    <div class="input-group">
                        <label>Font Style</label>
                        <select id="fontStyle" onchange="changeFontStyle()">
                            <option value="normal">Normal</option>
                            <option value="monospace">Monospace (Code)</option>
                        </select>
                    </div>

                    <div class="toolbar">
                        <button onclick="insertText('**bold**')"><b>B</b></button>
                        <button onclick="insertText('*italic*')"><i>I</i></button>
                        <button onclick="insertText('# ')">H1</button>
                        <button onclick="insertText('## ')">H2</button>
                        <button onclick="insertText('- ')">• List</button>
                        <button onclick="insertText('1. ')">1. List</button>
                        <button onclick="insertText('```\n\n```')">Code</button>
                        <button onclick="insertText('[link](url)')">🔗 Link</button>
                        <button onclick="insertTimestamp()">🕐 Time</button>
                        <button onclick="insertText('> ')">Quote</button>
                    </div>

                    <div class="input-group">
                        <label for="content">Content</label>
                        <textarea id="content" placeholder="Start typing your content here..."></textarea>
                    </div>

                    <div class="button-group">
                        <button class="primary" onclick="downloadFile()">💾 Download</button>
                        <button class="secondary" onclick="saveToMemory()">💼 Save to Memory</button>
                        <button class="secondary" onclick="downloadAllAsZip()">📦 Export All as ZIP</button>
                        <button class="secondary" onclick="clearContent()">🗑️ Clear</button>
                    </div>

                    <div class="success-message" id="successMessage"></div>
                </div>

                <div class="sidebar">
                    <h3>📊 Statistics</h3>
                    <div class="char-count">
                        <div><strong>Characters:</strong> <span id="charCount">0</span></div>
                        <div><strong>Words:</strong> <span id="wordCount">0</span></div>
                        <div><strong>Lines:</strong> <span id="lineCount">0</span></div>
                        <div><strong>Format:</strong> <span id="currentFormat">TXT</span></div>
                    </div>

                    <h3>💼 Saved Files (<span id="fileCount">0</span>)</h3>
                    <div class="saved-files" id="savedFiles">
                        <p style="color: #999; font-size: 13px;">No files saved yet</p>
                    </div>
                    <button class="secondary" style="width: 100%; margin-top: 10px;" onclick="clearAllFiles()">🗑️ Clear All</button>
                </div>
            </div>

            <div class="info-box">
                <strong>🔒 Privacy:</strong> All files are stored locally in your browser. Nothing is uploaded to any server.
            </div>
        </div>

        <!-- Templates Tab -->
        <div id="templates-tab" class="tab-content">
            <h2 style="margin-bottom: 20px;">Quick Start Templates</h2>
            
            <div class="template-grid">
                <div class="template-card" onclick="loadTemplate('meeting')">
                    <h4>📅 Meeting Notes</h4>
                    <p>Template for meeting minutes and action items</p>
                </div>
                <div class="template-card" onclick="loadTemplate('todo')">
                    <h4>✅ To-Do List</h4>
                    <p>Organized task list with priorities</p>
                </div>
                <div class="template-card" onclick="loadTemplate('blog')">
                    <h4>✍️ Blog Post</h4>
                    <p>Markdown blog post structure</p>
                </div>
                <div class="template-card" onclick="loadTemplate('readme')">
                    <h4>📖 README</h4>
                    <p>Project README template</p>
                </div>
                <div class="template-card" onclick="loadTemplate('code')">
                    <h4>💻 Code Snippet</h4>
                    <p>Code documentation template</p>
                </div>
                <div class="template-card" onclick="loadTemplate('journal')">
                    <h4>📔 Journal Entry</h4>
                    <p>Daily journal template</p>
                </div>
            </div>
        </div>

        <!-- Batch Tab -->
        <div id="batch-tab" class="tab-content">
            <h2 style="margin-bottom: 20px;">Create Multiple Files at Once</h2>
            
            <div class="input-group">
                <label>Enter each file content separated by "---" with optional filename</label>
                <p style="font-size: 13px; color: #666; margin-bottom: 10px;">Format: filename.txt<br>content<br>---</p>
                <textarea id="batchContent" style="min-height: 300px;" placeholder="file1.txt
Hello World
---
file2.txt
Second file content
---
file3.txt
Third file content"></textarea>
            </div>

            <div class="button-group">
                <button class="primary" onclick="downloadBatch()">📦 Download All Files</button>
                <button class="secondary" onclick="downloadBatchAsZip()">🗜️ Download as ZIP</button>
                <button class="secondary" onclick="clearBatch()">🗑️ Clear</button>
            </div>

            <div class="success-message" id="batchSuccessMessage"></div>
        </div>

        <!-- Import Tab -->
        <div id="import-tab" class="tab-content">
            <h2 style="margin-bottom: 20px;">Import Files</h2>
            
            <div class="drop-zone" id="dropZone">
                <p style="font-size: 18px; margin-bottom: 10px;">🗂️ Drag & Drop Files Here</p>
                <p>or</p>
                <input type="file" id="fileInput" multiple style="display: none;" onchange="handleFiles(this.files)">
                <button class="secondary" style="margin-top: 15px;" onclick="document.getElementById('fileInput').click()">
                    📁 Browse Files
                </button>
                <p style="margin-top: 15px; font-size: 12px; color: #999;">Supports: .txt, .md, .html, .css, .js, .json, .csv, .xml</p>
            </div>

            <div id="importedFiles"></div>
        </div>

        <!-- Netlify Guide Tab -->
        <div id="guide-tab" class="tab-content">
            <h2 style="margin-bottom: 20px;">🚀 Deploy to Netlify - Simple Guide</h2>

            <div class="guide-section">
                <h3>Method 1: Drag & Drop (Easiest)</h3>
                
                <div class="guide-step">
                    <strong>Step 1: Save this file</strong>
                    <p>Right-click anywhere on this page → "Save As" → Save as <code>index.html</code></p>
                </div>

                <div class="guide-step">
                    <strong>Step 2: Go to Netlify</strong>
                    <p>Visit <a href="https://www.netlify.com/" target="_blank" style="color: #667eea;">netlify.com</a> and sign up for free (GitHub/Email)</p>
                </div>

                <div class="guide-step">
                    <strong>Step 3: Deploy</strong>
                    <p>1. Click "Add new site" → "Deploy manually"<br>
                    2. Drag your <code>index.html</code> file into the drop zone<br>
                    3. Wait 10 seconds - Done! 🎉</p>
                </div>

                <div class="guide-step">
                    <strong>Step 4: Get your URL</strong>
                    <p>You'll get a URL like: <code>https://random-name-123.netlify.app</code><br>
                    You can customize this in Site Settings → Change site name</p>
                </div>
            </div>

            <div class="guide-section">
                <h3>Method 2: GitHub (For Version Control)</h3>
                
                <div class="guide-step">
                    <strong>Step 1: Create GitHub Repository</strong>
                    <p>1. Go to <a href="https://github.com/new" target="_blank" style="color: #667eea;">github.com/new</a><br>
                    2. Name it "text-file-creator"<br>
                    3. Make it Public<br>
                    4. Click "Create repository"</p>
                </div>

                <div class="guide-step">
                    <strong>Step 2: Upload the file</strong>
                    <p>1. Click "uploading an existing file"<br>
                    2. Drag your <code>index.html</code> file<br>
                    3. Click "Commit changes"</p>
                </div>

                <div class="guide-step">
                    <strong>Step 3: Connect to Netlify</strong>
                    <p>1. In Netlify, click "Add new site" → "Import an existing project"<br>
                    2. Choose GitHub<br>
                    3. Select your repository<br>
                    4. Click "Deploy site"</p>
                </div>

                <div class="guide-step">
                    <strong>✨ Bonus: Now updates are automatic!</strong>
                    <p>Every time you update the file in GitHub, Netlify will auto-deploy changes.</p>
                </div>
            </div>

            <div class="info-box">
                <strong>💡 Pro Tips:</strong><br>
                • The free tier is perfect for this app<br>
                • Your app will have HTTPS automatically<br>
                • You can add a custom domain later<br>
                • All data still stays in users' browsers (private)
            </div>
        </div>
    </div>

    <script>
        let currentFormat = 'txt';
        let savedFiles = JSON.parse(localStorage.getItem('savedFiles')) || {};

        const content = document.getElementById('content');
        const filename = document.getElementById('filename');
        const charCount = document.getElementById('charCount');
        const wordCount = document.getElementById('wordCount');
        const lineCount = document.getElementById('lineCount');
        const successMessage = document.getElementById('successMessage');

        // Update statistics
        content.addEventListener('input', updateStats);

        function updateStats() {
            const text = content.value;
            charCount.textContent = text.length;
            
            const words = text.trim().split(/\s+/).filter(word => word.length > 0);
            wordCount.textContent = words.length;
            
            const lines = text.split('\n').length;
            lineCount.textContent = lines;
        }

        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(tabName + '-tab').classList.add('active');
        }

        function selectFormat(format) {
            currentFormat = format;
            document.querySelectorAll('.format-btn').forEach(btn => btn.classList.remove('selected'));
            event.target.classList.add('selected');
            document.getElementById('currentFormat').textContent = format.toUpperCase();
        }

        function changeFontStyle() {
            const style = document.getElementById('fontStyle').value;
            if (style === 'monospace') {
                content.classList.add('monospace');
            } else {
                content.classList.remove('monospace');
            }
        }

        function insertText(text) {
            const start = content.selectionStart;
            const end = content.selectionEnd;
            const value = content.value;
            
            content.value = value.substring(0, start) + text + value.substring(end);
            content.focus();
            content.selectionStart = content.selectionEnd = start + text.length;
            updateStats();
        }

        function insertTimestamp() {
            const now = new Date();
            const timestamp = now.toLocaleString();
            insertText(timestamp);
        }

        function downloadFile() {
            const text = content.value;
            const name = filename.value.trim() || 'untitled';
            
            if (text.trim() === '') {
                showMessage('Please enter some content first!', false);
                return;
            }

            const blob = new Blob([text], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${name}.${currentFormat}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            showMessage(`✓ File "${name}.${currentFormat}" downloaded!`, true);
        }

        function saveToMemory() {
            const text = content.value;
            const name = filename.value.trim() || 'untitled';
            
            if (text.trim() === '') {
                showMessage('Please enter some content first!', false);
                return;
            }

            const fileKey = `${name}.${currentFormat}`;
            savedFiles[fileKey] = text;
            localStorage.setItem('savedFiles', JSON.stringify(savedFiles));
            
            updateSavedFilesList();
            showMessage(`✓ Saved "${fileKey}" to memory!`, true);
        }

        function updateSavedFilesList() {
            const container = document.getElementById('savedFiles');
            const count = Object.keys(savedFiles).length;
            document.getElementById('fileCount').textContent = count;
            
            if (count === 0) {
                container.innerHTML = '<p style="color: #999; font-size: 13px;">No files saved yet</p>';
                return;
            }

            container.innerHTML = '';
            
            for (const [fileName, fileContent] of Object.entries(savedFiles)) {
                const item = document.createElement('div');
                item.className = 'file-item';
                item.innerHTML = `
                    <div class="file-item-name">${fileName}</div>
                    <div class="file-item-actions">
                        <button onclick="loadFile('${fileName}')">Load</button>
                        <button onclick="downloadSavedFile('${fileName}')" style="background: #4caf50;">↓</button>
                        <button class="delete" onclick="deleteFile('${fileName}')">✕</button>
                    </div>
                `;
                container.appendChild(item);
            }
        }

        function loadFile(fileName) {
            content.value = savedFiles[fileName];
            filename.value = fileName.substring(0, fileName.lastIndexOf('.'));
            const format = fileName.substring(fileName.lastIndexOf('.') + 1);
            
            currentFormat = format;
            updateStats();
            showMessage(`✓ Loaded "${fileName}"`, true);
            switchTab('editor');
        }

        function downloadSavedFile(fileName) {
            const text = savedFiles[fileName];
            const blob = new Blob([text], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

        function deleteFile(fileName) {
            if (confirm(`Delete "${fileName}"?`)) {
                delete savedFiles[fileName];
                localStorage.setItem('savedFiles', JSON.stringify(savedFiles));
                updateSavedFilesList();
                showMessage(`✓ Deleted "${fileName}"`, true);
            }
        }

        function clearAllFiles() {
            if (confirm('Delete ALL saved files? This cannot be undone.')) {
                savedFiles = {};
                localStorage.setItem('savedFiles', JSON.stringify(savedFiles));
                updateSavedFilesList();
                showMessage('✓ All files cleared', true);
            }
        }

        function clearContent() {
            if (content.value.trim() === '' || confirm('Clear all content?')) {
                content.value = '';
                updateStats();
            }
        }

        async function downloadAllAsZip() {
            if (Object.keys(savedFiles).length === 0) {
                showMessage('No files saved to export!', false);
                return;
            }

            const zip = new JSZip();
            
            for (const [fileName, fileContent] of Object.entries(savedFiles)) {
                zip.file(fileName, fileContent);
            }

            const blob = await zip.generateAsync({type: 'blob'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'text-
