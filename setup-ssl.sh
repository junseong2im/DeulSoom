#!/bin/bash
# Setup SSL certificates with Let's Encrypt

# Configuration
DOMAIN="fragrance-ai.com"
EMAIL="admin@fragrance-ai.com"

echo "Setting up SSL certificates for $DOMAIN"
echo "========================================"

# Create directories
mkdir -p certbot/conf certbot/www

# Get initial certificate
docker-compose -f docker-compose.prod.yml run --rm certbot certonly \
  --webroot \
  --webroot-path=/var/www/certbot \
  --email $EMAIL \
  --agree-tos \
  --no-eff-email \
  -d $DOMAIN \
  -d www.$DOMAIN

echo ""
echo "SSL certificate obtained successfully!"
echo "Reloading nginx..."

# Reload nginx
docker-compose -f docker-compose.prod.yml exec nginx nginx -s reload

echo ""
echo "Setup complete! Your site is now accessible via HTTPS."
echo "Certificate will auto-renew every 12 hours."
