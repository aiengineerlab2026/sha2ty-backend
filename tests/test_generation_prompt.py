from app.services.generation_prompt import MESHY_TEXTURE_PROMPT_MAX, astra_prompt, meshy_texture_prompt, meshy_prompt


def test_prompt_names_the_object_and_keeps_photo_colours():
    p = meshy_texture_prompt("ceiling_light", "Brass pendant")
    assert p.startswith("Carpenter-ready modular ceiling light only.")
    # Removal is the Image to Image step's job: Meshy uses texture_prompt for the texture only.
    assert "Remove" not in p and "reflective mirrors exactly as in the reference image" in p
    assert "ceiling light" in p and "Brass pendant" in p
    assert "Match the reference image's colors, patterns, textures, and materials" in p  # the brief's texture part
    assert "clear glass, and reflective mirrors" in p and "Remove" not in p
    assert "plain neutral background" in meshy_prompt("ceiling_light") and "ceiling light" in meshy_prompt("ceiling_light")
    assert len(p) <= MESHY_TEXTURE_PROMPT_MAX


def test_unknown_category_falls_back_to_furniture_and_long_titles_fit_meshy():
    assert "piece of furniture" in meshy_texture_prompt("spaceship")
    assert len(meshy_texture_prompt("tv", "x" * 2000)) <= MESHY_TEXTURE_PROMPT_MAX


def test_astra_prompt_still_asks_for_a_glb():
    p = astra_prompt("table", "Oak dining table")
    assert p.startswith("Create a high-quality, photorealistic, structurally accurate 3D reconstruction of the table shown in the reference image.")
    assert "Keep flat surfaces flat, straight edges straight" in p and "Strictly avoid: Geometry distortion" in p  # the whole brief
    assert "Glass stays transparent" in p and "Mirrors stay reflective" in p and "plain material colours" not in p
    assert "a single table" in p and "/mnt/data/model.glb" in p


def test_wardrobe_and_wall_art_prompts():
    assert "wardrobe" in meshy_texture_prompt("wardrobe")
    assert "framed picture" in astra_prompt("wall_art", "Sunset print")


def test_prompts_focus_on_the_product_and_ignore_its_surroundings():
    prompt = astra_prompt("furniture", "Oslo sofa")
    assert "one object of this type" in prompt and "any other furniture or objects" in prompt and "plants" in prompt
    assert "chairs at it" not in prompt
    image = meshy_prompt("furniture", "sofa")
    assert image.startswith("Create a clean, isolated product reference image of the sofa shown.")
    assert "Remove all objects placed on, inside, or around the sofa" in image


def test_a_table_keeps_its_chairs_and_nothing_else():
    prompt = astra_prompt("table", "Oak dining set")
    assert "the table and the chairs at it, nothing else" in prompt
    assert "any other furniture or objects" in prompt and "tableware" in prompt
    image = meshy_prompt("table")
    assert "the table shown" in image and "plain neutral background" in image and "three-quarter product view" in image
    assert len(meshy_texture_prompt("table", "x" * 80)) <= MESHY_TEXTURE_PROMPT_MAX
