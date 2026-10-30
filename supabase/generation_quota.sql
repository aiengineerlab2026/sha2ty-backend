-- 3D generation quota: each vendor gets a fixed number of Meshy and GPT Astra calls
-- (GENERATION_LIMIT, default 30 each). Only the backend (service role) reads or changes it.

create table if not exists public.generation_usage (
  vendor_id uuid not null references auth.users (id) on delete cascade,
  provider text not null check (provider in ('meshy', 'astra')),
  used integer not null default 0 check (used >= 0),
  primary key (vendor_id, provider)
);
alter table public.generation_usage enable row level security;  -- no policies: service role only
revoke all on public.generation_usage from anon, authenticated;

-- Atomically take one try (delta 1) or give one back (delta -1). Returns tries used after the
-- change, or null when taking a try would go over the limit (nothing changes then).
create or replace function public.use_generation(v uuid, p text, delta integer, max_calls integer) returns integer
  language sql volatile security definer set search_path = public
  as $$
    insert into public.generation_usage as g (vendor_id, provider, used)
    values (v, p, greatest(delta, 0))
    on conflict (vendor_id, provider) do update set used = greatest(g.used + delta, 0)
    where g.used + delta <= max_calls or delta < 0
    returning used
  $$;
revoke execute on function public.use_generation(uuid, text, integer, integer) from public, anon, authenticated;

notify pgrst, 'reload schema';
