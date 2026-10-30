-- Prices for models, paints / floor finishes and carpets: a total price per item, or a price per m².
-- In a plan, paints are charged by the area they cover; models and carpets by the piece
-- (a per-m² model or carpet is charged for its own footprint). Safe to run more than once.
do $$
declare t text;
begin
  foreach t in array array['models', 'paints', 'carpets'] loop
    execute format('alter table public.%I add column if not exists price numeric check (price >= 0)', t);
    execute format('alter table public.%I add column if not exists price_unit text not null default %L', t, case when t = 'paints' then 'm2' else 'item' end);
    execute format('alter table public.%I drop constraint if exists %I', t, t || '_price_unit_check');
    execute format('alter table public.%I add constraint %I check (price_unit in (''item'', ''m2''))', t, t || '_price_unit_check');
  end loop;
end $$;

notify pgrst, 'reload schema';
