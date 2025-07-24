# GenomeVault Efficiency-Accuracy Trade-off Demo

A single-page interactive demo showing how users can adjust the computational efficiency vs. accuracy trade-off in GenomeVault queries.

## Key Concept

**Privacy is ALWAYS maintained** - your genomic data remains cryptographically protected at all accuracy levels. The trade-off is between:
- **Lower accuracy** = Faster queries, lower compute cost, but may need multiple iterations
- **Higher accuracy** = Slower queries, higher compute cost, but fewer iterations needed

## Features

- **Interactive Slider**: Adjust accuracy from 50% to 99%
- **Real-time Stats**: See how compute cost, query time, and iterations change
- **Privacy Indicator**: Constant reminder that privacy is always protected
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
2. **Efficiency Calculation**: Lower accuracy = less computational resources needed
3. **Iterative Refinement**: Start with fast, low-accuracy queries and re-run as needed
4. **Privacy Guarantee**: Cryptographic protection remains constant at all levels
5. **Results Display**: Shows how many iterations might be needed for convergence

## The Innovation

Traditional privacy-preserving systems force you to choose between privacy and accuracy. GenomeVault breaks this paradigm:
- Privacy is **always on** through hyperdimensional computing and zero-knowledge proofs
- You only trade off computational efficiency for accuracy
- Any uncertainty can be resolved by running the pipeline multiple times
- Start fast and refine iteratively - perfect for exploratory analysis

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
