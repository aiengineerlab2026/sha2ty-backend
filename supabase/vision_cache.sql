-- Vision model answers, so the same image is never sent to the model twice (saves tokens).
-- key = sha256 of image bytes + model + prompt/schema version (+ the vendor's category for products).
-- kind: 'floor_plan' (Workflow A) or 'product' (Workflow B, vendor photo before 3D generation).
create table if not exists public.vision_cache (
  key text primary key,
  kind text not null,
  model text not null,
  result jsonb not null,
  created_at timestamptz not null default now(),
  hits integer not null default 0
);

-- No policies: only the service role (the API) can read or write.
alter table public.vision_cache enable row level security;
