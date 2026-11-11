-- Open / closed states of one product (wardrobes first). The row itself is the default, closed state (glb_url,
-- source_image_url, components, material_edits stay as they are, so single-state models are unchanged); other
-- states live here, keyed by name:
--   {"open": {"glb_url": "...", "source_image_url": "...", "components": {...}, "material_edits": {...}}}
-- One product, one row: a state is never a second product. Safe to run more than once.

alter table public.models add column if not exists states jsonb;
alter table public.user_models add column if not exists states jsonb;

alter table public.models drop constraint if exists models_states_check;
alter table public.models add constraint models_states_check
  check (states is null or (jsonb_typeof(states) = 'object' and (not states ? 'open' or states -> 'open' ? 'glb_url')));
alter table public.user_models drop constraint if exists user_models_states_check;
alter table public.user_models add constraint user_models_states_check
  check (states is null or (jsonb_typeof(states) = 'object' and (not states ? 'open' or states -> 'open' ? 'glb_url')));

notify pgrst, 'reload schema';
