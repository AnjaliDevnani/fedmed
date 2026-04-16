const fs = require('fs');
const css = fs.readFileSync('style.css', 'utf8');
const js = fs.readFileSync('script.js', 'utf8');
const shell = fs.readFileSync('view_shell.html', 'utf8');
const dashboard = fs.readFileSync('view_dashboard.html', 'utf8');
const upload = fs.readFileSync('view_upload.html', 'utf8');
const monitor = fs.readFileSync('view_monitor.html', 'utf8');
const predictions = fs.readFileSync('view_predictions.html', 'utf8');
const architecture = fs.readFileSync('view_architecture.html', 'utf8');

const htmlBody = shell.replace('<main class="main" id="main-content"></main>', 
  '<main class="main" id="main-content"></main>\n' + dashboard + '\n' + upload + '\n' + monitor + '\n' + predictions + '\n' + architecture
);

const finalHtml = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FedMed \u2014 Federated Learning Platform</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>${css}</style>
</head>
<body>
${htmlBody}
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
  <script>${js}</script>
</body>
</html>`;

fs.writeFileSync('index.html', finalHtml);
console.log('Successfully built index.html!');
