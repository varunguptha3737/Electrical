from PIL import Image

from visual_agent.tools.vision import encode_image, image_block


def test_encode_image_downscales():
    img = Image.new("RGB", (4000, 2000), "red")
    url = encode_image(img, max_edge=1000)
    assert url.startswith("data:image/png;base64,")
    # decode back and check size
    import base64, io

    data = base64.b64decode(url.split(",", 1)[1])
    out = Image.open(io.BytesIO(data))
    assert max(out.size) == 1000
    assert out.size == (1000, 500)


def test_image_block_shape():
    img = Image.new("RGB", (10, 10))
    block = image_block(img)
    assert block["type"] == "image_url"
    assert block["image_url"]["url"].startswith("data:image/png;base64,")
