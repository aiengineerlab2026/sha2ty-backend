-- Store model dimensions in meters.

do $$
begin
  if exists (
    select 1 from information_schema.columns
    where table_schema = 'public' and table_name = 'models' and column_name = 'width_cm'
  ) then
    alter table public.models rename column width_cm to width_m;
    alter table public.models rename column height_cm to height_m;
    alter table public.models rename column depth_cm to depth_m;
    update public.models set
      width_m = width_m / 100.0
    where width_m is not null and width_m >= 1;
    update public.models set
      height_m = height_m / 100.0
    where height_m is not null and height_m >= 1;
    update public.models set
      depth_m = depth_m / 100.0
    where depth_m is not null and depth_m >= 1;
  end if;
end $$;

notify pgrst, 'reload schema';
