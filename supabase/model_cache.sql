-- Finished 3D results, so the same image is not converted (vision model, Meshy / Astra) twice.
-- kind 'floor_plan_3d': the converted plan JSON, key = sha256(image + wall height/thickness + vision model + pipeline version).
-- kind 'vendor_glb': a vendor's generated model {glb_url, details}, key = sha256(image + vendor + provider + category + title + texture).
create table if not exists public.model_cache (
  key text primary key,
  kind text not null,
  model text not null,
  result jsonb not null,
  created_at timestamptz not null default now()
);

-- No policies: only the service role (the API) can read or write.
alter table public.model_cache enable row level security;
