# Shirt Reel Maker

Mockup থেকে ১০ সেকেন্ডের 9:16 reel ভিডিও বানানোর টুল (ব্রাউজারেই চলে, সার্ভারে কিছু আপলোড হয় না)।

- `site/index.html` — পুরো টুল, একটাই ফাইল। NAS-এ এটাই বসানো হয়।
- `docker-compose.yml` + `nginx.conf` + `entry.sh` — NAS-এ nginx দিয়ে চালানোর জন্য (`docker compose up -d`; http পোর্ট 8089, https পোর্ট 8443)।
- `src/` + `build.py` — সোর্স। বদলালে `python3 build.py` চালালে `site/index.html` আবার তৈরি হয়।
- `README-bangla.md` — NAS-এ বসানো ও আপডেট করার বিস্তারিত নিয়ম।

আপডেট: নতুন `site/index.html` দিয়ে NAS-এর `site/index.html` বদলে দিন, তারপর ব্রাউজারে রিফ্রেশ।


## Video template tab

The **Video template** tab puts your PNG designs onto a real shirt video (hands, zoom and camera moves included) and makes one 10 second reel per design, as a ZIP. The shipped template lives in `site/templates/`. To add more, see `tools/video_template/README.md`.

## Image marketing tab

Puts PNG designs on shirt and mug photos and downloads them as JPEG (one by one, or all as a ZIP). The shared mockup library is the folder `site/mockups/` plus its `list.json` (`[{"file":"tee1.jpg","name":"White tee","kind":"shirt"},{"file":"mug1.jpg","name":"White mug","kind":"mug"}]`, kind is `shirt` or `mug`). People can also add their own mockup photos in the tab; those stay in the browser.
