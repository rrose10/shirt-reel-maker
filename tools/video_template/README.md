# Making a new video template

A template is a real video of a plain shirt (fixed camera, 9:16, about 10 s) where the printed design is
tracked and removed, so the tool can place any other design in every frame.

Work in an empty folder:

```
ffmpeg -i source.mp4 -vf scale=720:1280 fr/%03d.png      # mkdir fr first
python3 track.py         # tracks the old print in every frame  -> Ws.npy
python3 refine.py        # retries the weak frames              -> Ws2.npy
python3 build_tpl.py     # clean plates (cl/), hand masks, shade -> skins.npy, track_raw.json
python3 pack.py ../site/templates/my-template "My template name"
```
Then add `{ "id": "my-template", "name": "My template name" }` to `site/templates/list.json`.

Only use footage you filmed or have a licence for. The tracker follows a white print on a dark shirt.
