-- Migration for deployments that already have public.leak_logs.
-- Adds the idempotency key used to make leak logging retry-safe and
-- account-pinned. Safe to run once; re-running will error on the existing
-- constraint, which can be ignored.
alter table public.leak_logs
  add column if not exists leak_id text;

alter table public.leak_logs
  add constraint leak_logs_leak_id_key unique (leak_id);
