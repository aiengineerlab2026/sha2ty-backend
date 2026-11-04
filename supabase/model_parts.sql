-- Editable parts of a generated model, the vendor's material edits, and the untouched Meshy/Astra GLB.
-- glb_url becomes the customized file once the vendor saves an edit. Safe to run more than once.
alter table public.models add column if not exists components jsonb;
alter table public.models add column if not exists material_edits jsonb;
alter table public.models add column if not exists source_glb_url text;

notify pgrst, 'reload schema';
