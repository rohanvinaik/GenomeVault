# Recording the Efficiency-Accuracy Trade-off Demo GIF

## Quick Steps to Create the Demo GIF

### 1. Start the Web Server
```bash
cd examples/webdial
python3 -m http.server 8080
# Visit http://localhost:8080
```

### 2. Record with QuickTime (macOS)
1. Open QuickTime Player
2. File → New Screen Recording
3. Click the arrow next to record button → Select recording area
4. Select just the browser window showing the demo
5. Record these actions:
   - Move slider from 50% to 99% slowly
   - Show compute cost changing from "Very Low" to "High"
   - Show iterations needed changing from "3-5x" to "1x"
   - Note the privacy indicator stays constant
   - Click "Run Privacy-Preserving Query"
   - Show the loading animation
   - Display query results

### 3. Convert to GIF
Using ffmpeg:
```bash
# First convert .mov to .mp4 with good quality
ffmpeg -i demo_recording.mov -vcodec h264 -acodec mp2 demo_recording.mp4

# Then convert to GIF
ffmpeg -i demo_recording.mp4 -vf "fps=15,scale=600:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 accuracy-dial.gif

# Or use gifski for better quality
gifski -o accuracy-dial.gif demo_recording.mp4 --fps 15 --width 600
```

### 4. Optimize the GIF
```bash
# Use gifsicle to optimize
gifsicle -O3 --colors 128 accuracy-dial.gif -o accuracy-dial-optimized.gif
```

### 5. Upload to Repository
Place the final GIF in:
```
assets/demo/accuracy-dial.gif
```

## Alternative: Use Web-based Tools

1. **LICEcap** (Windows/macOS): https://www.cockos.com/licecap/
2. **Peek** (Linux): https://github.com/phw/peek
3. **ScreenToGif** (Windows): https://www.screentogif.com/
4. **Kap** (macOS): https://getkap.co/

## Demo Script

For consistency, follow this script when recording:

1. **Start** - Show the demo at 85% accuracy
2. **Slide to 50%** - Show "Very Low" compute cost, "3-5x" iterations needed
3. **Highlight privacy box** - Emphasize that privacy remains constant
4. **Slide to 99%** - Show "High" compute cost, "1x" iteration  
5. **Click Query Button** - Show loading spinner
6. **Show Results** - Display JSON response
7. **Slide back to 85%** - Return to default

Key message: "Start fast with low accuracy, refine as needed - privacy always protected!"

Total duration: ~10-15 seconds

## Tips for a Great GIF

- Keep it under 5MB for GitHub
- Use a clean browser (hide bookmarks, extensions)
- Record at 2x retina resolution, then scale down
- Ensure smooth cursor movements
- Add a 1-2 second pause at start/end
