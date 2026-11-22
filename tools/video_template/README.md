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

## Shirts with a busy background (mannequin / shop)
Use `track_roi.py` and `build_roi.py` instead of `track.py` / `refine.py` / `build_tpl.py`:

```
python3 track_roi.py REF_FRAME x0,y0,x1,y1 [BRIGHT_THRESHOLD]   # a frame where the whole print is visible, and a box around it
python3 build_roi.py [REMOVE_THRESHOLD]
python3 pack.py ../site/templates/my-template "My template name"
```

## Blank-shirt variant (fixed camera, nothing to remove)
When the source video shows a BLANK shirt and the camera does not move, there is no print to track or erase:
1. Extract frames at 720x1280 into `fr/NNN.png`.
2. `python3 build_blank.py FPS x,y,w,h` (x,y,w,h = print area in frame pixels) – writes identity homographies, skin masks and a shade map.
3. `python3 pack.py OUTDIR "Name"` – packs template.mp4 + track.json (fps is taken from track_raw.json).
4. Add `{id,name}` to site/templates/list.json.

Blank-shirt templates use a two-sided shade map (`"bi":1` in track.json): code 128 = neutral, below = fold in shadow, above = fold catching light (code = 128 + (brightness/blur - 1)*254). The tool darkens/brightens the print with it and bends the print slightly along the folds ("Follow fabric folds" slider).
