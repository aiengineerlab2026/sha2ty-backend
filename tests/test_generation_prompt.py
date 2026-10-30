from app.services.generation_prompt import MESHY_TEXTURE_PROMPT_MAX, astra_prompt, meshy_texture_prompt


def test_prompt_names_the_object_and_keeps_photo_colours():
    p = meshy_texture_prompt("ceiling_light", "Brass pendant")
    assert "ceiling light" in p and "Brass pendant" in p
    assert "exact colours" in p and "Ignore the background" in p
    assert len(p) <= MESHY_TEXTURE_PROMPT_MAX


def test_unknown_category_falls_back_to_furniture_and_long_titles_fit_meshy():
    assert "piece of furniture" in meshy_texture_prompt("spaceship")
    assert len(meshy_texture_prompt("tv", "x" * 2000)) <= MESHY_TEXTURE_PROMPT_MAX


def test_astra_prompt_still_asks_for_a_glb():
    p = astra_prompt("table", "Oak dining table")
    assert "a single table" in p and "/mnt/data/model.glb" in p and "exact colours" in p


def test_wardrobe_and_wall_art_prompts():
    assert "wardrobe" in meshy_texture_prompt("wardrobe")
    assert "framed picture" in astra_prompt("wall_art", "Sunset print")
