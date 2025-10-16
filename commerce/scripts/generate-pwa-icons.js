/**
 * PWA Icon Generator Script
 *
 * This script generates PWA icons from a source image.
 *
 * Usage:
 *   node scripts/generate-pwa-icons.js <source-image-path>
 *
 * Requirements:
 *   npm install sharp
 *
 * Generates:
 *   - icon-192x192.png (required for PWA)
 *   - icon-512x512.png (required for PWA)
 *   - icon-96x96.png (for shortcuts)
 *   - favicon.ico (for browsers)
 */

const fs = require('fs');
const path = require('path');

async function generateIcons(sourcePath) {
  try {
    // Check if sharp is installed
    let sharp;
    try {
      sharp = require('sharp');
    } catch (e) {
      console.error('Error: sharp is not installed.');
      console.error('Please install it by running: npm install sharp');
      process.exit(1);
    }

    const publicDir = path.join(__dirname, '..', 'public');

    // Check if source file exists
    if (!fs.existsSync(sourcePath)) {
      console.error(`Error: Source file not found: ${sourcePath}`);
      process.exit(1);
    }

    console.log('Generating PWA icons...');
    console.log(`Source: ${sourcePath}`);

    // Generate icons
    const sizes = [
      { size: 192, name: 'icon-192x192.png' },
      { size: 512, name: 'icon-512x512.png' },
      { size: 96, name: 'icon-96x96.png' }
    ];

    for (const { size, name } of sizes) {
      const outputPath = path.join(publicDir, name);
      await sharp(sourcePath)
        .resize(size, size, {
          fit: 'cover',
          position: 'center'
        })
        .png()
        .toFile(outputPath);
      console.log(`✓ Generated: ${name} (${size}x${size})`);
    }

    // Generate favicon
    const faviconPath = path.join(publicDir, 'favicon.ico');
    await sharp(sourcePath)
      .resize(32, 32)
      .toFile(faviconPath);
    console.log('✓ Generated: favicon.ico (32x32)');

    console.log('\nSuccess! All PWA icons generated.');
    console.log('Icons are saved in:', publicDir);
  } catch (error) {
    console.error('Error generating icons:', error.message);
    process.exit(1);
  }
}

// Get source image from command line argument
const sourceImage = process.argv[2];

if (!sourceImage) {
  console.log('PWA Icon Generator');
  console.log('==================\n');
  console.log('Usage: node scripts/generate-pwa-icons.js <source-image-path>');
  console.log('\nExample:');
  console.log('  node scripts/generate-pwa-icons.js ./logo.png');
  console.log('\nRequired sizes:');
  console.log('  - 192x192 (PWA manifest)');
  console.log('  - 512x512 (PWA manifest)');
  console.log('  - 96x96 (shortcuts)');
  console.log('  - 32x32 (favicon)');
  console.log('\nMake sure to install sharp first:');
  console.log('  npm install sharp');
  process.exit(1);
}

generateIcons(sourceImage);
