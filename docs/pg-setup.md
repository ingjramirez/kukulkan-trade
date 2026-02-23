# PostgreSQL Setup (Hetzner Server)

One-time steps to migrate production from SQLite to PostgreSQL.

## 1. Install PostgreSQL

```bash
apt update && apt install -y postgresql postgresql-contrib
systemctl enable postgresql
systemctl start postgresql
```

## 2. Create Database & User

```bash
sudo -u postgres psql <<SQL
CREATE USER kukulkan WITH PASSWORD 'CHANGE_ME_STRONG_PASSWORD';
CREATE DATABASE kukulkan OWNER kukulkan;
GRANT ALL PRIVILEGES ON DATABASE kukulkan TO kukulkan;
SQL
```

## 3. Update .env

Add to `/opt/kukulkan-trade/.env`:

```
DATABASE_URL=postgresql+asyncpg://kukulkan:CHANGE_ME_STRONG_PASSWORD@localhost:5432/kukulkan
```

## 4. Install Python PG Driver

```bash
cd /opt/kukulkan-trade
source .venv/bin/activate
pip install -e .          # asyncpg is now in dependencies
pip install psycopg2-binary  # needed for migration scripts only
```

## 5. Run PG Baseline Migration

```bash
python scripts/migrate.py --db postgresql://kukulkan:PASS@localhost:5432/kukulkan
```

This creates all 26 tables + the default tenant.

## 6. Migrate Data from SQLite

```bash
python scripts/migrate_data.py \
    --sqlite data/kukulkan.db \
    --pg postgresql://kukulkan:PASS@localhost:5432/kukulkan
```

Verify row counts match expectations. The script is idempotent (safe to re-run).

## 7. Restart Services

```bash
systemctl restart kukulkan-bot
systemctl restart kukulkan-api
```

Both services read `DATABASE_URL` from `.env` via pydantic-settings.

## 8. Verify

```bash
# Check API responds
curl -s http://localhost:8001/api/health | jq .

# Check PG directly
sudo -u postgres psql kukulkan -c "SELECT count(*) FROM tenants;"
sudo -u postgres psql kukulkan -c "SELECT count(*) FROM trades;"
```

## Rollback

If something goes wrong, remove `DATABASE_URL` from `.env` and restart services.
They will fall back to SQLite at `data/kukulkan.db` (the default).
