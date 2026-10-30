-- The catalogue (paints / floor finishes / carpets) is public: anyone can browse it, signed in or not.
-- Writes stay owner-only. Models are served publicly by the backend catalog API (ready models only).
grant select on public.paints, public.carpets to anon;

drop policy if exists paints_read on public.paints;
create policy paints_read on public.paints for select to anon, authenticated using (true);

drop policy if exists carpets_read on public.carpets;
create policy carpets_read on public.carpets for select to anon, authenticated using (true);

notify pgrst, 'reload schema';
