/**
 * Recognition Ledger Widget
 * A lightweight, customizable widget for displaying Recognition Science predictions
 * 
 * Usage:
 *   <div id="recognition-ledger"></div>
 *   <script src="https://cdn.jsdelivr.net/gh/jonwashburn/recognition-ledger@main/widget.js"></script>
 */

(function() {
  'use strict';

  // Default configuration
  const defaultConfig = {
    containerId: 'recognition-ledger',
    theme: 'light', // light, dark, minimal, scientific
    showSummary: true,
    showPredictions: true,
    maxPredictions: 10,
    autoUpdate: false,
    updateInterval: 3600000, // 1 hour
    baseUrl: 'https://raw.githubusercontent.com/jonwashburn/recognition-ledger/main'
  };

  // Core widget class
  class RecognitionLedgerWidget {
    constructor(config = {}) {
      this.config = { ...defaultConfig, ...config };
      this.container = document.getElementById(this.config.containerId);
      
      if (!this.container) {
        console.error(`Container with id "${this.config.containerId}" not found`);
        return;
      }

      this.predictions = [];
      this.init();
    }

    async init() {
      this.addStyles();
      this.showLoading();
      
      try {
        await this.loadPredictions();
        this.render();
        
        if (this.config.autoUpdate) {
          this.startAutoUpdate();
        }
      } catch (error) {
        this.showError(error);
      }
    }

    addStyles() {
      if (document.getElementById('recognition-ledger-styles')) return;

      const styles = document.createElement('style');
      styles.id = 'recognition-ledger-styles';
      styles.textContent = `
        .rl-widget {
          font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          line-height: 1.6;
          color: #333;
          background: #fff;
          border-radius: 8px;
          padding: 20px;
          box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .rl-widget.dark {
          background: #1a1a1a;
          color: #e0e0e0;
          box-shadow: 0 2px 10px rgba(0,0,0,0.5);
        }

        .rl-widget.minimal {
          box-shadow: none;
          border: 1px solid #e0e0e0;
        }

        .rl-widget.scientific {
          font-family: 'Computer Modern', Georgia, serif;
          background: #fafafa;
        }

        .rl-header {
          text-align: center;
          margin-bottom: 20px;
        }

        .rl-title {
          font-size: 1.5em;
          margin: 0 0 10px 0;
          color: inherit;
        }

        .rl-subtitle {
          font-size: 0.9em;
          opacity: 0.8;
        }

        .rl-summary {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 15px;
          margin: 20px 0;
        }

        .rl-stat {
          text-align: center;
          padding: 15px;
          background: rgba(0,0,0,0.05);
          border-radius: 6px;
        }

        .dark .rl-stat {
          background: rgba(255,255,255,0.1);
        }

        .rl-stat-number {
          display: block;
          font-size: 2em;
          font-weight: bold;
          color: #2196F3;
        }

        .dark .rl-stat-number {
          color: #64B5F6;
        }

        .rl-stat-label {
          display: block;
          font-size: 0.85em;
          margin-top: 5px;
          opacity: 0.8;
        }

        .rl-predictions {
          margin: 20px 0;
        }

        .rl-prediction {
          border: 1px solid #e0e0e0;
          border-radius: 6px;
          padding: 12px;
          margin: 10px 0;
          transition: all 0.3s ease;
        }

        .dark .rl-prediction {
          border-color: #444;
        }

        .rl-prediction:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .rl-prediction.verified {
          border-color: #4CAF50;
          background: rgba(76, 175, 80, 0.05);
        }

        .rl-prediction.pending {
          border-color: #FF9800;
          background: rgba(255, 152, 0, 0.05);
        }

        .rl-prediction.refuted {
          border-color: #F44336;
          background: rgba(244, 67, 54, 0.05);
        }

        .rl-prediction-header {
          display: flex;
          align-items: center;
          margin-bottom: 8px;
        }

        .rl-status-icon {
          font-size: 1.2em;
          margin-right: 8px;
        }

        .rl-prediction-title {
          font-weight: 600;
          flex: 1;
        }

        .rl-values {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 10px;
          font-size: 0.9em;
          color: #666;
        }

        .dark .rl-values {
          color: #aaa;
        }

        .rl-accuracy {
          margin-top: 8px;
          font-size: 0.85em;
          font-weight: bold;
          color: #4CAF50;
        }

        .rl-footer {
          text-align: center;
          margin-top: 20px;
          padding-top: 20px;
          border-top: 1px solid #e0e0e0;
        }

        .dark .rl-footer {
          border-color: #444;
        }

        .rl-button {
          display: inline-block;
          padding: 10px 20px;
          background: #2196F3;
          color: white;
          text-decoration: none;
          border-radius: 5px;
          transition: background 0.3s ease;
        }

        .rl-button:hover {
          background: #1976D2;
        }

        .rl-loading, .rl-error {
          text-align: center;
          padding: 40px;
        }

        .rl-spinner {
          display: inline-block;
          width: 40px;
          height: 40px;
          border: 3px solid #f3f3f3;
          border-top: 3px solid #2196F3;
          border-radius: 50%;
          animation: rl-spin 1s linear infinite;
        }

        @keyframes rl-spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        @media (max-width: 600px) {
          .rl-widget {
            padding: 15px;
          }
          
          .rl-summary {
            grid-template-columns: 1fr;
          }
          
          .rl-values {
            grid-template-columns: 1fr;
          }
        }
      `;

      document.head.appendChild(styles);
    }

    showLoading() {
      this.container.innerHTML = `
        <div class="rl-widget ${this.config.theme}">
          <div class="rl-loading">
            <div class="rl-spinner"></div>
            <p>Loading Recognition Science predictions...</p>
          </div>
        </div>
      `;
    }

    showError(error) {
      this.container.innerHTML = `
        <div class="rl-widget ${this.config.theme}">
          <div class="rl-error">
            <p>Unable to load predictions. Please try again later.</p>
            <a href="https://github.com/jonwashburn/recognition-ledger" class="rl-button">
              View on GitHub
            </a>
          </div>
        </div>
      `;
      console.error('Recognition Ledger Widget Error:', error);
    }

    async loadPredictions() {
      // List of core predictions to load
      const predictionFiles = [
        'electron_mass',
        'muon_mass',
        'tau_mass',
        'fine_structure',
        'gravitational_constant',
        'dark_energy',
        'hubble_constant'
      ];

      const promises = predictionFiles.map(file =>
        fetch(`${this.config.baseUrl}/predictions/${file}.json`)
          .then(res => res.json())
          .catch(err => {
            console.warn(`Failed to load ${file}:`, err);
            return null;
          })
      );

      const results = await Promise.all(promises);
      this.predictions = results.filter(p => p !== null);
    }

    render() {
      const verified = this.predictions.filter(p => p.verification.status === 'verified').length;
      const total = this.predictions.length;
      const accuracy = verified / total * 100;

      let html = `
        <div class="rl-widget ${this.config.theme}">
          <div class="rl-header">
            <h2 class="rl-title">Recognition Science Ledger</h2>
            <p class="rl-subtitle">Zero-parameter unified physics</p>
          </div>
      `;

      if (this.config.showSummary) {
        html += `
          <div class="rl-summary">
            <div class="rl-stat">
              <span class="rl-stat-number">${verified}</span>
              <span class="rl-stat-label">Verified</span>
            </div>
            <div class="rl-stat">
              <span class="rl-stat-number">${total}</span>
              <span class="rl-stat-label">Predictions</span>
            </div>
            <div class="rl-stat">
              <span class="rl-stat-number">0</span>
              <span class="rl-stat-label">Free Parameters</span>
            </div>
            <div class="rl-stat">
              <span class="rl-stat-number">${accuracy.toFixed(0)}%</span>
              <span class="rl-stat-label">Accuracy</span>
            </div>
          </div>
        `;
      }

      if (this.config.showPredictions) {
        const predictionsToShow = this.predictions.slice(0, this.config.maxPredictions);
        
        html += '<div class="rl-predictions">';
        predictionsToShow.forEach(pred => {
          html += this.renderPrediction(pred);
        });
        html += '</div>';
      }

      html += `
        <div class="rl-footer">
          <a href="https://github.com/jonwashburn/recognition-ledger" 
             class="rl-button" 
             target="_blank" 
             rel="noopener noreferrer">
            View Full Ledger →
          </a>
        </div>
      </div>
      `;

      this.container.innerHTML = html;
    }

    renderPrediction(pred) {
      const statusIcons = {
        'verified': '✓',
        'pending': '⏳',
        'refuted': '✗'
      };

      const icon = statusIcons[pred.verification.status] || '?';
      const predicted = pred.prediction.value;
      const measured = pred.verification.measurement.value;
      const deviation = Math.abs((predicted - measured) / measured * 100);

      return `
        <div class="rl-prediction ${pred.verification.status}">
          <div class="rl-prediction-header">
            <span class="rl-status-icon">${icon}</span>
            <span class="rl-prediction-title">${pred.prediction.observable}</span>
          </div>
          <div class="rl-values">
            <div>Predicted: ${predicted.toExponential(4)} ${pred.prediction.unit}</div>
            <div>Measured: ${measured.toExponential(4)} ${pred.prediction.unit}</div>
          </div>
          ${deviation < 1 ? `<div class="rl-accuracy">Accuracy: ${(100 - deviation).toFixed(3)}%</div>` : ''}
        </div>
      `;
    }

    startAutoUpdate() {
      setInterval(() => {
        this.loadPredictions().then(() => this.render());
      }, this.config.updateInterval);
    }
  }

  // Auto-initialize if default container exists
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      if (document.getElementById('recognition-ledger')) {
        new RecognitionLedgerWidget();
      }
    });
  } else {
    if (document.getElementById('recognition-ledger')) {
      new RecognitionLedgerWidget();
    }
  }

  // Expose to global scope for manual initialization
  window.RecognitionLedgerWidget = RecognitionLedgerWidget;
})(); 