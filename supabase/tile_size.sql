-- Texture tile size as width × height in metres (one image tile covers this much wall / floor).
-- Existing rows inherit the old square tile_m. Safe to run more than once.
alter table public.paints add column if not exists tile_width_m numeric check (tile_width_m > 0);
alter table public.paints add column if not exists tile_height_m numeric check (tile_height_m > 0);
update public.paints set tile_width_m = coalesce(tile_width_m, tile_m), tile_height_m = coalesce(tile_height_m, tile_m)
  where tile_width_m is null or tile_height_m is null;

notify pgrst, 'reload schema';
