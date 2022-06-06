import pickle
import mplcursors
import matplotlib.pyplot as plt
plt.figure()
plt.close()
model_type = 'tile'
fig = pickle.load(open(f'D:/applied_models/{model_type}/tSNE.pkl', 'rb'))
mplcursors.cursor(fig).connect("add", lambda sel: sel.annotation.set_text(
    sel.artist.annots[sel.target.index]))
mplcursors.cursor(fig, hover=True).connect("add",
                                           lambda sel: sel.annotation.set_text(sel.artist.im_paths[sel.target.index]))
plt.show()
