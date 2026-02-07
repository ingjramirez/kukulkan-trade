# Kukulkan Infrastructure Guide

## Architecture

```
Internet → Cloudflare (DNS + SSL) → Hetzner VPS (nginx :80)
                                        ├── kukulkan.trade      → /var/www/kukulkan.trade/index.html
                                        └── app.kukulkan.trade  → reverse proxy → Streamlit :8501
```

- **Cloudflare** handles DNS and SSL termination (free tier)
- **Nginx** serves the static landing page and proxies the dashboard
- **Streamlit** runs as a systemd service bound to `127.0.0.1:8501`
- **Basic auth** protects the dashboard (`/etc/nginx/.htpasswd-kukulkan`)

## Files

| File | Purpose |
|------|---------|
| `deploy/landing/index.html` | Landing page served at kukulkan.trade |
| `deploy/nginx/kukulkan.trade` | Nginx config: landing page + www redirect |
| `deploy/nginx/app.kukulkan.trade` | Nginx config: dashboard reverse proxy with basic auth |
| `deploy/kukulkan-dashboard.service` | Systemd unit for Streamlit dashboard |

## Server Setup (one-time)

Run these commands on the Hetzner VPS as root:

```bash
# 1. Install nginx and htpasswd tool
apt-get update && apt-get install -y nginx apache2-utils

# 2. Deploy landing page
mkdir -p /var/www/kukulkan.trade
cp /opt/kukulkan-trade/deploy/landing/index.html /var/www/kukulkan.trade/

# 3. Install nginx configs
cp /opt/kukulkan-trade/deploy/nginx/kukulkan.trade /etc/nginx/sites-available/
cp /opt/kukulkan-trade/deploy/nginx/app.kukulkan.trade /etc/nginx/sites-available/
ln -sf /etc/nginx/sites-available/kukulkan.trade /etc/nginx/sites-enabled/
ln -sf /etc/nginx/sites-available/app.kukulkan.trade /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# 4. Create dashboard password
PASS=$(openssl rand -base64 18)
htpasswd -cb /etc/nginx/.htpasswd-kukulkan admin "$PASS"
echo "Dashboard password: $PASS"   # Save this!

# 5. Test and reload nginx
nginx -t && systemctl enable nginx && systemctl restart nginx

# 6. Install and start dashboard service
cp /opt/kukulkan-trade/deploy/kukulkan-dashboard.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable kukulkan-dashboard
systemctl start kukulkan-dashboard

# 7. Verify
curl -s -o /dev/null -w "%{http_code}" http://localhost/        # Should be 200
curl -s -o /dev/null -w "%{http_code}" http://localhost:8501/_stcore/health  # Should be "ok"
```

## Cloudflare DNS Setup (manual)

1. Add domain `kukulkan.trade` to Cloudflare (free plan)
2. Create DNS records:

| Type | Name | Content | Proxy |
|------|------|---------|-------|
| A | `@` | `128.140.102.191` | Proxied |
| A | `app` | `128.140.102.191` | Proxied |
| CNAME | `www` | `kukulkan.trade` | Proxied |

3. SSL/TLS settings:
   - Encryption mode: **Full** (not Full Strict — no origin cert needed)
   - Always Use HTTPS: **On**
   - Automatic HTTPS Rewrites: **On**

4. Page Rules (optional):
   - `www.kukulkan.trade/*` → Forwarding URL (301) → `https://kukulkan.trade/$1`

## CI/CD

The GitHub Actions workflow (`deploy.yml`) automatically:
1. Rsyncs the landing page to `/var/www/kukulkan.trade/`
2. Copies nginx configs and reloads nginx
3. Restarts `kukulkan-dashboard` service

## Updating the landing page

Just edit `deploy/landing/index.html` and push to `main`. CI/CD handles deployment.

## Updating dashboard password

```bash
ssh root@128.140.102.191
htpasswd -b /etc/nginx/.htpasswd-kukulkan admin NEW_PASSWORD
systemctl reload nginx
```

## Troubleshooting

```bash
# Check nginx
nginx -t
journalctl -u nginx -f

# Check dashboard
systemctl status kukulkan-dashboard
journalctl -u kukulkan-dashboard -f

# Check if Streamlit is listening
ss -tlnp | grep 8501
```
