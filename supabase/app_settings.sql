-- Site-wide settings the admin changes at runtime (read by the API with the service role key).
-- vision_model: the model that reads images into JSON (floor plans, vendor product photos).
create table if not exists public.app_settings (
  key text primary key,
  value jsonb not null,
  updated_at timestamptz not null default now()
);

-- No policies: only the service role (the API) can read or write.
alter table public.app_settings enable row level security;

insert into public.app_settings (key, value) values ('vision_model', '"gemini-3.1-pro-preview"')
on conflict (key) do nothing;
