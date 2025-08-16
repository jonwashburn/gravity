# API Integration Guide

> Quick setup instructions for displaying Recognition Ledger data on any website, with specific examples for RecognitionJournal.org

## 🚀 Quick Start (5 minutes)

### 1. Direct GitHub Data Access

The simplest way to display Recognition Ledger data on your website:

```html
<!-- Add this to your HTML -->
<div id="recognition-predictions"></div>

<script>
// Fetch predictions directly from GitHub
fetch('https://raw.githubusercontent.com/jonwashburn/recognition-ledger/main/predictions/electron_mass.json')
  .then(response => response.json())
  .then(data => {
    document.getElementById('recognition-predictions').innerHTML = `
      <h3>${data.prediction.observable}</h3>
      <p>Predicted: ${data.prediction.value} ${data.prediction.unit}</p>
      <p>Status: <span class="status-${data.verification.status}">${data.verification.status}</span></p>
    `;
  });
</script>

<style>
.status-verified { color: green; font-weight: bold; }
.status-pending { color: orange; }
.status-refuted { color: red; }
</style>
```

### 2. For RecognitionJournal.org

Here's a complete widget you can drop into any page:

```javascript
// Recognition Ledger Widget
class RecognitionWidget {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.baseUrl = 'https://raw.githubusercontent.com/jonwashburn/recognition-ledger/main';
  }

  async loadPredictions() {
    const predictions = [
      'electron_mass',
      'muon_mass',
      'fine_structure',
      'gravitational_constant'
    ];

    const results = await Promise.all(
      predictions.map(p => 
        fetch(`${this.baseUrl}/predictions/${p}.json`)
          .then(r => r.json())
          .catch(() => null)
      )
    );

    this.render(results.filter(r => r !== null));
  }

  render(predictions) {
    const verified = predictions.filter(p => p.verification.status === 'verified').length;
    const total = predictions.length;

    this.container.innerHTML = `
      <div class="recognition-widget">
        <h2>Recognition Science Status</h2>
        <div class="summary">
          <div class="stat">
            <span class="number">${verified}</span>
            <span class="label">Verified</span>
          </div>
          <div class="stat">
            <span class="number">${total}</span>
            <span class="label">Total Predictions</span>
          </div>
          <div class="stat">
            <span class="number">0</span>
            <span class="label">Free Parameters</span>
          </div>
        </div>
        <div class="predictions">
          ${predictions.map(p => this.renderPrediction(p)).join('')}
        </div>
        <a href="https://github.com/jonwashburn/recognition-ledger" class="view-all">
          View Full Ledger →
        </a>
      </div>
    `;
  }

  renderPrediction(pred) {
    const deviation = pred.verification.deviation_sigma || 0;
    const statusIcon = {
      'verified': '✓',
      'pending': '⏳',
      'refuted': '✗'
    }[pred.verification.status];

    return `
      <div class="prediction ${pred.verification.status}">
        <h3>${statusIcon} ${pred.prediction.observable}</h3>
        <div class="values">
          <span>Predicted: ${pred.prediction.value.toFixed(9)}</span>
          <span>Measured: ${pred.verification.measurement.value.toFixed(9)}</span>
        </div>
        <div class="accuracy">Accuracy: ${(1 - Math.abs(deviation)).toFixed(6)}</div>
      </div>
    `;
  }
}

// Initialize widget
document.addEventListener('DOMContentLoaded', () => {
  const widget = new RecognitionWidget('recognition-dashboard');
  widget.loadPredictions();
});
```

### 3. Complete Integration Example

Create this as `recognition-feed.html` on RecognitionJournal.org:

```html
<!DOCTYPE html>
<html>
<head>
  <title>Recognition Science Live Feed</title>
  <style>
    .recognition-widget {
      font-family: system-ui, -apple-system, sans-serif;
      max-width: 800px;
      margin: 0 auto;
      padding: 20px;
    }
    
    .summary {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 20px;
      margin: 20px 0;
    }
    
    .stat {
      text-align: center;
      padding: 20px;
      background: #f5f5f5;
      border-radius: 8px;
    }
    
    .stat .number {
      display: block;
      font-size: 2em;
      font-weight: bold;
      color: #333;
    }
    
    .stat .label {
      display: block;
      margin-top: 5px;
      color: #666;
    }
    
    .prediction {
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 15px;
      margin: 10px 0;
      background: #fff;
    }
    
    .prediction.verified {
      border-color: #4CAF50;
      background: #f1f8f1;
    }
    
    .prediction.pending {
      border-color: #FF9800;
      background: #fff8f1;
    }
    
    .prediction h3 {
      margin: 0 0 10px 0;
      color: #333;
    }
    
    .values {
      display: flex;
      justify-content: space-between;
      font-size: 0.9em;
      color: #666;
    }
    
    .accuracy {
      margin-top: 5px;
      font-size: 0.85em;
      color: #4CAF50;
      font-weight: bold;
    }
    
    .view-all {
      display: inline-block;
      margin-top: 20px;
      padding: 10px 20px;
      background: #2196F3;
      color: white;
      text-decoration: none;
      border-radius: 5px;
    }
    
    .view-all:hover {
      background: #1976D2;
    }
  </style>
</head>
<body>
  <div id="recognition-dashboard"></div>
  
  <script>
    // Paste the RecognitionWidget class here
    // (from section 2 above)
  </script>
</body>
</html>
```

## 📊 Advanced Integration

### Fetching All Predictions

```javascript
async function getAllPredictions() {
  // First, get the directory listing
  const response = await fetch(
    'https://api.github.com/repos/jonwashburn/recognition-ledger/contents/predictions'
  );
  const files = await response.json();
  
  // Filter for JSON files
  const jsonFiles = files.filter(f => f.name.endsWith('.json'));
  
  // Fetch all predictions
  const predictions = await Promise.all(
    jsonFiles.map(async (file) => {
      const data = await fetch(file.download_url).then(r => r.json());
      return data;
    })
  );
  
  return predictions;
}
```

### Real-time Updates via GitHub API

```javascript
// Check for updates every hour
setInterval(async () => {
  const response = await fetch(
    'https://api.github.com/repos/jonwashburn/recognition-ledger/commits?path=predictions&per_page=1'
  );
  const [latestCommit] = await response.json();
  
  if (latestCommit.sha !== localStorage.getItem('lastCommitSha')) {
    localStorage.setItem('lastCommitSha', latestCommit.sha);
    // Reload predictions
    widget.loadPredictions();
    console.log('New predictions available!');
  }
}, 3600000); // Check every hour
```

### CORS-Free Proxy Options

If you encounter CORS issues:

1. **Use JSDelivr CDN** (recommended):
   ```javascript
   const baseUrl = 'https://cdn.jsdelivr.net/gh/jonwashburn/recognition-ledger@main';
   ```

2. **GitHub Pages** (if enabled):
   ```javascript
   const baseUrl = 'https://jonwashburn.github.io/recognition-ledger';
   ```

3. **Your own proxy**:
   ```javascript
   const baseUrl = 'https://recognitionjournal.org/api/ledger-proxy';
   ```

## 🔧 For WordPress Sites

Create a plugin file `recognition-ledger-widget.php`:

```php
<?php
/*
Plugin Name: Recognition Ledger Widget
Description: Display Recognition Science predictions
*/

function recognition_ledger_shortcode() {
    return '<div id="recognition-dashboard"></div>
    <script src="https://cdn.jsdelivr.net/gh/jonwashburn/recognition-ledger@main/widget.js"></script>';
}

add_shortcode('recognition_ledger', 'recognition_ledger_shortcode');
```

Then use `[recognition_ledger]` anywhere in your content.

## 🎨 Customization

### Themes

```css
/* Dark theme */
.recognition-widget.dark {
  background: #1a1a1a;
  color: #fff;
}

/* Minimal theme */
.recognition-widget.minimal .prediction {
  border: none;
  border-bottom: 1px solid #eee;
}

/* Scientific theme */
.recognition-widget.scientific {
  font-family: 'Computer Modern', Georgia, serif;
}
```

### Custom Filtering

```javascript
// Show only verified predictions
const verified = predictions.filter(p => p.verification.status === 'verified');

// Show only high-precision predictions
const highPrecision = predictions.filter(p => 
  p.prediction.uncertainty < 1e-6
);

// Sort by deviation
predictions.sort((a, b) => 
  Math.abs(a.verification.deviation_sigma) - Math.abs(b.verification.deviation_sigma)
);
```

## 📱 Mobile Responsive

The widget is mobile-friendly by default. For custom layouts:

```css
@media (max-width: 600px) {
  .summary {
    grid-template-columns: 1fr;
  }
  
  .values {
    flex-direction: column;
  }
}
```

## 🚨 Error Handling

```javascript
class RobustRecognitionWidget extends RecognitionWidget {
  async loadPredictions() {
    try {
      // ... loading code ...
    } catch (error) {
      this.container.innerHTML = `
        <div class="error">
          <p>Unable to load predictions. Please try again later.</p>
          <a href="https://github.com/jonwashburn/recognition-ledger">
            View on GitHub
          </a>
        </div>
      `;
    }
  }
}
```

## 🔐 API Rate Limits

GitHub API allows 60 requests/hour unauthenticated. To increase:

```javascript
const headers = {
  'Authorization': 'token YOUR_GITHUB_TOKEN'
};

fetch(url, { headers })
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/jonwashburn/recognition-ledger/issues)
- **Email**: jon@recognitionphysics.org
- **Documentation**: This file

## 🎯 Next Steps

1. Copy the Quick Start code
2. Customize the styling to match your site
3. Add to your website
4. Monitor the console for any errors
5. Share your implementation!

---

*Last updated: January 2025* 