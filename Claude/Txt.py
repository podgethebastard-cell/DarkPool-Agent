#!/usr/bin/env python3
# TextFile.py - Python script to generate the HTML

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
        /* ... rest of your code ... */
    </style>
</head>
<body>
    <div class="container">
        <!-- Your HTML content -->
    </div>
    <script>
        // Your JavaScript code
    </script>
</body>
</html>'''

# Write to file
with open('index.html', 'w') as f:
    f.write(html_content)

print("HTML file generated successfully as 'index.html'")
