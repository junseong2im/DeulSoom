/**
 * SVG to PNG Converter for PWA Icons
 *
 * Converts the icon.svg to required PNG sizes
 *
 * Usage:
 *   npm install sharp
 *   node scripts/svg-to-png.js
 */

const fs = require('fs');
const path = require('path');

async function convertSvgToPng() {
  try {
    // Try to load sharp
    let sharp;
    try {
      sharp = require('sharp');
    } catch (e) {
      console.error('Error: sharp is not installed.');
      console.error('Install it by running: npm install sharp');
      process.exit(1);
    }

    const publicDir = path.join(__dirname, '..', 'public');
    const svgPath = path.join(publicDir, 'icon.svg');

    // Check if SVG exists
    if (!fs.existsSync(svgPath)) {
      console.error('Error: icon.svg not found in public directory');
      console.error('Please create icon.svg first');
      process.exit(1);
    }

    console.log('Converting icon.svg to PNG formats...\n');

    // Read SVG
    const svgBuffer = fs.readFileSync(svgPath);

    // Define sizes
    const sizes = [
      { size: 192, name: 'icon-192x192.png', desc: 'PWA manifest' },
      { size: 512, name: 'icon-512x512.png', desc: 'PWA manifest' },
      { size: 96, name: 'icon-96x96.png', desc: 'Shortcuts' },
      { size: 32, name: 'favicon.ico', desc: 'Browser favicon' },
      { size: 180, name: 'apple-touch-icon.png', desc: 'iOS home screen' }
    ];

    // Convert each size
    for (const { size, name, desc } of sizes) {
      const outputPath = path.join(publicDir, name);

      await sharp(svgBuffer)
        .resize(size, size, {
          fit: 'contain',
          background: { r: 0, g: 0, b: 0, alpha: 0 }
        })
        .png()
        .toFile(outputPath);

      console.log(`✓ ${name.padEnd(25)} ${size}x${size} - ${desc}`);
    }

    console.log('\n✅ All icons generated successfully!');
    console.log(`📁 Location: ${publicDir}`);

  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
}

// Run conversion
convertSvgToPng();
