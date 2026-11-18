# Shirt Reel Maker

Mockup থেকে ১০ সেকেন্ডের 9:16 reel ভিডিও বানানোর টুল (ব্রাউজারেই চলে, সার্ভারে কিছু আপলোড হয় না)।

- `site/index.html` — পুরো টুল, একটাই ফাইল। NAS-এ এটাই বসানো হয়।
- `docker-compose.yml` + `nginx.conf` — NAS-এ nginx দিয়ে চালানোর জন্য (`docker compose up -d`, পোর্ট 8089)।
- `src/` + `build.py` — সোর্স। বদলালে `python3 build.py` চালালে `site/index.html` আবার তৈরি হয়।
- `README-bangla.md` — NAS-এ বসানো ও আপডেট করার বিস্তারিত নিয়ম।

আপডেট: নতুন `site/index.html` দিয়ে NAS-এর `site/index.html` বদলে দিন, তারপর ব্রাউজারে রিফ্রেশ।
