# Dead by Daylight Logo Setup

The website now displays the Dead by Daylight logo automatically!

## How It Works

The logo display has a 3-tier fallback system:

1. **Local Logo (Priority 1)**: If you place `dbd_logo.png` in the `static/` folder, it will be used
2. **Online Logo (Priority 2)**: If no local logo is found, it uses an online CDN source
3. **Text Fallback (Priority 3)**: If both images fail, it shows "DEAD BY DAYLIGHT" text

## Option 1: Use Online Logo (Default - Already Working!)

The logo is already displayed using an online source. No action needed! Just run your Flask app and you'll see the official Dead by Daylight logo.

## Option 2: Use Your Own Logo File

If you want to use a local logo file instead:

1. **Download the Official Logo**
   - Visit: https://deadbydaylight.com/media/ (official media kit)
   - Or download from: https://prilogo.com/logo/dead-by-daylight/
   - Save as a PNG file with transparent background

2. **Save the Logo**
   - Rename your logo file to: `dbd_logo.png`
   - Place it in the `static/` folder:
   ```
   static/
   └── dbd_logo.png
   ```

3. **Restart Flask App**
   - The local logo will automatically be used instead of the online version

## Logo Requirements

- **Format**: PNG (preferred) or JPG
- **Size**: Recommended 300-600px wide (will be auto-resized)
- **Background**: Transparent PNG works best with the dark theme
- **Name**: Must be exactly `dbd_logo.png`

## Current Logo Status

✅ The website currently uses the online Dead by Daylight logo from logos-world.net
✅ The logo has a red glow effect that matches the horror theme
✅ The logo is responsive and works on all screen sizes

No additional setup required - the logo should already be visible when you run the app!

