# GenomeVault Accuracy Dial Demo

A single-page interactive demo showing how users can adjust the privacy vs. accuracy trade-off in GenomeVault queries.

## Features

- **Interactive Slider**: Adjust accuracy from 50% to 99%
- **Real-time Stats**: See how privacy level, query time, and data points change
- **Live API Integration**: Connects to GenomeVault's `/estimate_budget` and `/query_tuned` endpoints
- **Responsive Design**: Works on desktop and mobile devices
- **Zero Dependencies**: Pure HTML/CSS/JavaScript, no frameworks required

## Usage

### Option 1: Direct Browser
Simply open `index.html` in your web browser:
```bash
open index.html
```

### Option 2: With Local Server
For full API integration, serve with a local server:
```bash
# Using Python
python3 -m http.server 8080

# Using Node.js
npx http-server -p 8080

# Then visit http://localhost:8080
```

### Option 3: With GenomeVault API
1. Start the GenomeVault API server:
   ```bash
   cd ../..  # Navigate to genomevault root
   python -m genomevault.api.server
   ```

2. Update the `API_BASE` constant in `index.html` if needed

3. Open the demo in your browser

## How It Works

1. **Accuracy Slider**: Users adjust the desired accuracy level (50-99%)
2. **Privacy Calculation**: Higher accuracy = lower privacy (more data exposed)
3. **Budget Estimation**: Calls `/estimate_budget` to calculate privacy budget
4. **Query Execution**: Runs a tuned query with the calculated budget
5. **Results Display**: Shows variant frequencies with applied differential privacy

## API Endpoints Used

- `POST /api/v1/estimate_budget`: Calculates privacy budget for target accuracy
- `POST /api/v1/query_tuned`: Executes query with specified privacy budget

## Customization

Edit these variables in `index.html`:
- `API_BASE`: Your GenomeVault API endpoint
- Slider range: Adjust min/max accuracy values
- Visual theme: Modify CSS variables for colors

## Demo Mode

If the API is not available, the demo falls back to mock data that simulates realistic responses.

## Screenshot Usage

Perfect for README GIFs - the visual feedback and smooth animations make great demos!

```bash
# Record with QuickTime or similar
# 1. Open index.html
# 2. Adjust slider to show privacy trade-offs
# 3. Click "Run Query" to demonstrate
```
