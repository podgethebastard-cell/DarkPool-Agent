#!/usr/bin/env python3
# text_file_creator.py - Ultimate Text File Creator HTML Generator

import os
import sys

def create_html_file():
    """Generate the Ultimate Text File Creator HTML file"""
    
    html_content = '''<!DOCTYPE html>
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
                        <button onclick="insertText('```\\n\\n```')">Code</button>
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
            
            const words = text.trim().split(/\\s+/).filter(word => word.length > 0);
            wordCount.textContent = words.length;
            
            const lines = text.split('\\n').length;
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
            a.download = 'text-files.zip';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

        function showMessage(message, isSuccess) {
            successMessage.textContent = message;
            successMessage.style.display = 'block';
            successMessage.style.background = isSuccess ? '#4caf50' : '#f44336';
            
            setTimeout(() => {
                successMessage.style.display = 'none';
            }, 3000);
        }

        function loadTemplate(templateName) {
            const templates = {
                meeting: `# Meeting Notes

Date: ${new Date().toLocaleDateString()}
Time: ${new Date().toLocaleTimeString()}
Attendees: 

## Agenda
1. 
2. 
3. 

## Discussion Points
- 

## Action Items
- [ ] 
- [ ] 
- [ ] 

## Next Meeting
Date: 
Time: 
Topics: `,

                todo: `# To-Do List

## High Priority ⚠️
- [ ] 
- [ ] 
- [ ] 

## Medium Priority 📝
- [ ] 
- [ ] 

## Low Priority 💤
- [ ] 
- [ ] 

## Completed ✓
- [x] 

## Notes
`,
                
                blog: `# Blog Post Title

*By Your Name • ${new Date().toLocaleDateString()}*

## Introduction


## Main Content


## Conclusion


## Tags
#blog #writing #topic

---

*Thanks for reading!*`,

                readme: `# Project Name

Brief description of your project.

## Features
- Feature 1
- Feature 2
- Feature 3

## Installation

\`\`\`bash
# Installation commands
npm install
\`\`\`

## Usage

\`\`\`javascript
// Example code
console.log('Hello World');
\`\`\`

## Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| option1 | Description | value1 |
| option2 | Description | value2 |

## Contributing
Please read CONTRIBUTING.md for details.

## License
MIT License`,

                code: `# Code Documentation

/**
 * Function: functionName
 * Description: 
 * Parameters:
 *   - param1: 
 *   - param2: 
 * Returns: 
 * 
 * Example:
 *   functionName(arg1, arg2)
 */

\`\`\`language
// Code snippet here
function example() {
    return 'Hello World';
}
\`\`\`

## Usage Examples

## Dependencies

## Troubleshooting`,

                journal: `# Journal Entry - ${new Date().toLocaleDateString()}

## Morning
- 

## Afternoon
- 

## Evening
- 

## Thoughts & Reflections
- 

## Goals for Tomorrow
1. 
2. 
3. 

## Gratitude
- 
- 
- 

Mood: 😊 😐 😢 😡
Energy: 🔋 🔋 🔋 🔋`
            };

            if (templates[templateName]) {
                content.value = templates[templateName];
                filename.value = `${templateName}-${new Date().toISOString().split('T')[0]}`;
                updateStats();
                switchTab('editor');
                showMessage(`✓ Loaded ${templateName} template!`, true);
            }
        }

        function downloadBatch() {
            const batchText = document.getElementById('batchContent').value;
            const files = batchText.split('---').map(f => f.trim()).filter(f => f);
            
            if (files.length === 0) {
                showMessage('Please enter some content!', false);
                return;
            }

            files.forEach((fileContent, index) => {
                const lines = fileContent.split('\\n');
                let fileName = `file${index + 1}.txt`;
                let content = fileContent;
                
                if (lines.length > 1) {
                    const firstLine = lines[0].trim();
                    if (firstLine.includes('.') && firstLine.split('.')[1].match(/^[a-z0-9]+$/i)) {
                        fileName = firstLine;
                        content = lines.slice(1).join('\\n');
                    }
                }
                
                const blob = new Blob([content], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = fileName;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            });
            
            showMessage(`✓ Downloaded ${files.length} file(s)!`, true);
        }

        async function downloadBatchAsZip() {
            const batchText = document.getElementById('batchContent').value;
            const files = batchText.split('---').map(f => f.trim()).filter(f => f);
            
            if (files.length === 0) {
                showMessage('Please enter some content!', false);
                return;
            }

            const zip = new JSZip();
            
            files.forEach((fileContent, index) => {
                const lines = fileContent.split('\\n');
                let fileName = `file${index + 1}.txt`;
                let content = fileContent;
                
                if (lines.length > 1) {
                    const firstLine = lines[0].trim();
                    if (firstLine.includes('.') && firstLine.split('.')[1].match(/^[a-z0-9]+$/i)) {
                        fileName = firstLine;
                        content = lines.slice(1).join('\\n');
                    }
                }
                
                zip.file(fileName, content);
            });

            const blob = await zip.generateAsync({type: 'blob'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'batch-files.zip';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            showMessage(`✓ Created ZIP with ${files.length} file(s)!`, true);
        }

        function clearBatch() {
            if (confirm('Clear all batch content?')) {
                document.getElementById('batchContent').value = '';
            }
        }

        function handleFiles(files) {
            const dropZone = document.getElementById('dropZone');
            dropZone.classList.remove('drag-over');
            
            const importedFilesContainer = document.getElementById('importedFiles');
            importedFilesContainer.innerHTML = '';
            
            Array.from(files).forEach(file => {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const item = document.createElement('div');
                    item.className = 'file-item';
                    item.innerHTML = `
                        <div class="file-item-name">${file.name}</div>
                        <div class="file-item-actions">
                            <button onclick="useImportedFile('${file.name}', \`${e.target.result.replace(/`/g, '\\`').replace(/\\$/g, '\\\\$')}\`)">Use</button>
                            <button onclick="downloadImportedFile('${file.name}', \`${e.target.result.replace(/`/g, '\\`').replace(/\\$/g, '\\\\$')}\`)">↓</button>
                        </div>
                    `;
                    importedFilesContainer.appendChild(item);
                };
                reader.readAsText(file);
            });
        }

        function useImportedFile(fileName, contentText) {
            document.getElementById('content').value = contentText;
            document.getElementById('filename').value = fileName.substring(0, fileName.lastIndexOf('.'));
            
            const ext = fileName.substring(fileName.lastIndexOf('.') + 1);
            if (['txt', 'md', 'html', 'csv', 'json', 'xml'].includes(ext)) {
                currentFormat = ext;
                document.querySelectorAll('.format-btn').forEach(btn => btn.classList.remove('selected'));
                document.querySelectorAll('.format-btn').forEach(btn => {
                    if (btn.textContent.includes(ext.toUpperCase())) {
                        btn.classList.add('selected');
                    }
                });
                document.getElementById('currentFormat').textContent = ext.toUpperCase();
            }
            
            updateStats();
            switchTab('editor');
            showMessage(`✓ Loaded ${fileName}`, true);
        }

        function downloadImportedFile(fileName, contentText) {
            const blob = new Blob([contentText], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

        // Initialize
        updateStats();
        updateSavedFilesList();

        // Drag and drop events
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');

        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            handleFiles(e.dataTransfer.files);
        });
    </script>
</body>
</html>'''

    return html_content


def save_html_file(filename="index.html"):
    """Save the HTML content to a file"""
    try:
        html_content = create_html_file()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Successfully created '{filename}'")
        print(f"📁 File size: {len(html_content)} characters")
        print("\n🎯 Next steps:")
        print("   1. Open this file in your web browser")
        print("   2. To deploy to Netlify, see the 'Netlify Guide' tab in the app")
        print("   3. Enjoy creating text files!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating file: {e}")
        return False


def create_python_web_server(port=8000):
    """Create a simple Python web server to serve the file"""
    import http.server
    import socketserver
    import webbrowser
    import threading
    
    # First save the file
    if not save_html_file():
        return
    
    print(f"\n🌐 Starting web server on http://localhost:{port}")
    print("   Press Ctrl+C to stop the server\n")
    
    # Change to current directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create handler
    handler = http.server.SimpleHTTPRequestHandler
    
    # Start server in a separate thread
    def start_server():
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"✅ Server running at http://localhost:{port}")
            httpd.serve_forever()
    
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Open browser
    webbrowser.open(f"http://localhost:{port}")
    
    try:
        # Keep main thread alive
        while True:
            server_thread.join(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")


def main():
    """Main function with command line interface"""
    print("=" * 50)
    print("📝 Ultimate Text File Creator - HTML Generator")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "serve" or command == "server":
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
            create_python_web_server(port)
        elif command == "save":
            filename = sys.argv[2] if len(sys.argv) > 2 else "index.html"
            save_html_file(filename)
        elif command == "help" or command == "--help":
            show_help()
        else:
            print(f"Unknown command: {command}")
            show_help()
    else:
        # Interactive mode
        print("\nChoose an option:")
        print("  1. Create HTML file only")
        print("  2. Create HTML file and start web server")
        print("  3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            filename = input("Enter filename [index.html]: ").strip() or "index.html"
            save_html_file(filename)
        elif choice == "2":
            port_input = input("Enter port number [8000]: ").strip()
            port = int(port_input) if port_input else 8000
            create_python_web_server(port)
        elif choice == "3":
            print("Goodbye!")
        else:
            print("Invalid choice")


def show_help():
    """Show help information"""
    print("\nUsage:")
    print("  python text_file_creator.py [command]")
    print("\nCommands:")
    print("  serve [port]   Create HTML and start web server (default port: 8000)")
    print("  save [name]    Save HTML file (default name: index.html)")
    print("  help           Show this help message")
    print("\nExamples:")
    print("  python text_file_creator.py serve       # Start server on port 8000")
    print("  python text_file_creator.py serve 8080  # Start server on port 8080")
    print("  python text_file_creator.py save        # Save as index.html")
    print("  python text_file_creator.py save myapp.html  # Save as myapp.html")


if __name__ == "__main__":
    main()
