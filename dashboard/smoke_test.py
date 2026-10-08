import time
from streamlit.testing.v1 import AppTest

t0 = time.time()
at = AppTest.from_file("app.py", default_timeout=180).run()
print(f"first run {time.time() - t0:.1f}s, exceptions: {[e.value for e in at.exception]}")
print("metrics:", [(m.label, m.value) for m in at.metric])

at.button[0].click().run()
print("after cutout click, exceptions:", [e.value for e in at.exception], "| warnings:", [w.value for w in at.warning])

at.multiselect[0].select("COVERCROPS").run()
print("covercrops filter, exceptions:", [e.value for e in at.exception], "| metrics:", [(m.label, m.value) for m in at.metric])

at.radio[0].set_value("Cutouts").run()
at.selectbox[0].set_value("Species and location").run()
print("coverage=cutouts + species/location grouping, exceptions:", [e.value for e in at.exception])
print("captions:", [c.value for c in at.caption][:6])
