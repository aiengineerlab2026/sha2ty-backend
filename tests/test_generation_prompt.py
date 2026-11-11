from app.services.generation_prompt import MESHY_TEXTURE_PROMPT_MAX, astra_prompt, meshy_texture_prompt, meshy_prompt


def test_prompt_names_the_object_and_keeps_photo_colours():
    p = meshy_texture_prompt("ceiling_light", "Brass pendant")
    assert "ceiling light" in p and "Brass pendant" in p
    assert "exact colours" in p
    assert "Ignore the background" in meshy_prompt("ceiling_light")  # what is modelled: the Image to Image prompt
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


def test_prompts_focus_on_the_product_and_ignore_its_surroundings():
    for prompt in (meshy_prompt("furniture", "sofa"), astra_prompt("furniture", "Oslo sofa")):
        assert "one object of this type" in prompt and "any other furniture or objects" in prompt and "plants" in prompt
        assert "chairs at it" not in prompt


def test_a_table_keeps_its_chairs_and_nothing_else():
    for prompt in (meshy_prompt("table"), astra_prompt("table", "Oak dining set")):
        assert "the table and the chairs at it, nothing else" in prompt
        assert "any other furniture or objects" in prompt and "tableware" in prompt
    assert len(meshy_texture_prompt("table", "x" * 80)) <= MESHY_TEXTURE_PROMPT_MAX
