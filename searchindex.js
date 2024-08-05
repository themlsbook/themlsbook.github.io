Search.setIndex({"docnames": ["chapter2/index", "chapter2/knn", "chapter3/cost_function", "chapter3/gradient_descent", "chapter3/index", "chapter3/linear_regression", "chapter4/basis_expansion", "chapter4/index", "chapter4/regularization", "chapter5/bias_variance_decomposition", "chapter5/index", "chapter5/validation_methods", "chapter6/filter_methods", "chapter6/index", "chapter6/search_methods", "chapter7/data_augmentation", "chapter7/data_cleaning", "chapter7/feature_transformation", "chapter7/index", "index"], "filenames": ["chapter2/index.md", "chapter2/knn.md", "chapter3/cost_function.md", "chapter3/gradient_descent.md", "chapter3/index.md", "chapter3/linear_regression.md", "chapter4/basis_expansion.md", "chapter4/index.md", "chapter4/regularization.md", "chapter5/bias_variance_decomposition.md", "chapter5/index.md", "chapter5/validation_methods.md", "chapter6/filter_methods.md", "chapter6/index.md", "chapter6/search_methods.md", "chapter7/data_augmentation.md", "chapter7/data_cleaning.md", "chapter7/feature_transformation.md", "chapter7/index.md", "index.md"], "titles": ["CHAPTER 2", "K-Nearest Neighbors Classifier", "Cost Function", "Gradient Descent", "CHAPTER 3", "Linear Regression", "Basis Expansion", "CHAPTER 4", "Regularization", "Bias-variance Decomposition", "CHAPTER 5", "Cross-validation Methods", "Filter Methods", "CHAPTER 6", "Search Methods", "Data Augmentation", "Data Cleaning", "Feature Transformation &amp; Binning", "CHAPTER 7", "About"], "terms": {"k": [0, 9, 16, 17], "nearest": [0, 17], "neighbor": 0, "classifi": [0, 14, 16], "www": [0, 4, 7, 10, 13, 18, 19], "themlsbook": [0, 4, 7, 10, 12, 13, 14, 18, 19], "com": [0, 4, 7, 10, 12, 13, 14, 18, 19], "thi": [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19], "supplement": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17, 19], "materi": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "machin": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17, 19], "simplifi": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17, 19], "book": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17, 19], "It": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "shed": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "light": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "python": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17, 19], "implement": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17, 19], "topic": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "discuss": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "while": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "all": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "detail": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "explan": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "can": [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17], "found": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "i": [1, 2, 3, 6, 8, 9, 11, 12, 14, 15, 16, 17], "also": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "assum": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "you": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "know": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "syntax": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "work": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "If": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "don": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "t": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "highli": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "recommend": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "take": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "break": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "get": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "introduc": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "languag": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "befor": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "go": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "forward": [1, 2, 3, 5, 6, 8, 9, 11, 12, 15, 16, 17], "my": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "code": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "download": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "jupyt": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "notebook": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "button": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "upper": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "right": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "corner": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "ipynb": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "reproduc": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "plai": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "around": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "we": [1, 2, 3, 5, 6, 7, 9, 11, 12, 15, 16, 17], "start": [1, 2, 3, 5, 6, 8], "need": [1, 2, 3, 5, 6, 15, 16], "import": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "few": [1, 2, 3, 5, 6, 15], "us": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "jupyterbook": [1, 2, 3, 5, 6], "understand": [1, 9], "what": [1, 3, 5, 11], "those": [1, 2], "do": [1, 5, 11, 15, 16, 17], "now": [1, 2, 3, 5, 9, 12], "panda": [1, 8, 9, 12, 14, 15, 16, 17], "pd": [1, 8, 9, 12, 14, 15, 16, 17], "numpi": [1, 2, 3, 5, 6, 8, 9, 11, 14, 17], "np": [1, 2, 3, 5, 6, 8, 9, 11, 14, 17], "from": [1, 2, 3, 5, 8, 9, 11, 12, 14, 15, 16, 17], "sklearn": [1, 5, 8, 11, 12, 14, 15, 17], "metric": [1, 14, 16], "kneighborsclassifi": 1, "matplotlib": [1, 2, 3, 5, 6, 8, 9], "pyplot": [1, 2, 3, 5, 6, 8, 9], "plt": [1, 2, 3, 5, 6, 8, 9], "color": [1, 2, 3, 5, 6, 8, 9], "listedcolormap": 1, "seaborn": 1, "sn": 1, "config": [1, 2, 3, 5, 6, 8, 9], "inlinebackend": [1, 2, 3, 5, 6, 8, 9], "figure_format": [1, 2, 3, 5, 6, 8, 9], "retina": [1, 2, 3, 5, 6, 8, 9], "sharper": [1, 2, 3, 5, 6, 8, 9], "scatter": [1, 6, 8], "graph": [1, 2, 3, 5, 6], "def": [1, 2, 6, 9], "plotfruitfigur": 1, "defin": [1, 2, 5, 6, 8, 12, 14, 15, 16, 17], "variabl": [1, 5, 9], "apple_height": 1, "apple_width": 1, "df": [1, 16, 17], "height": 1, "fruit": 1, "appl": 1, "width": 1, "mandarin_height": 1, "mandarin_width": 1, "mandarin": 1, "lemon_height": 1, "lemon_width": 1, "lemon": 1, "initi": [1, 5, 17], "fig": [1, 2, 3, 5], "ax": [1, 2, 3, 5, 9], "subplot": [1, 2, 3, 5], "gca": 1, "set_aspect": 1, "equal": [1, 9, 11], "adjust": 1, "box": 1, "o": [1, 2, 3, 5, 6, 8, 9], "r": [1, 3, 5, 9, 14], "label": [1, 2, 3, 5, 6, 8, 12, 14, 15], "g": [1, 2, 3, 5, 9], "b": [1, 2, 5, 9], "show": [1, 2, 3, 5, 6, 8, 9, 14, 16], "legend": [1, 2, 3, 5, 6, 8, 9], "configur": [1, 6, 8, 11], "s": [1, 2, 3, 5, 6, 7, 9, 11, 12, 14, 15, 16, 17, 19], "size": [1, 3, 5, 9, 11], "ylim": [1, 3, 5, 6, 8, 9], "10": [1, 2, 5, 6, 8, 9, 12, 14, 15, 17], "xlim": [1, 3, 5, 6, 8, 9], "11": [1, 5, 12, 14, 15], "titl": [1, 6, 9], "xlabel": 1, "ylabel": 1, "plotknn": 1, "n_neighbor": [1, 12], "int": [1, 12, 14], "plot_data": 1, "true": [1, 2, 3, 5, 8, 9, 12, 14, 16, 17], "plot_height": 1, "none": [1, 16, 17], "plot_width": 1, "plot_label": 1, "turn": 1, "categor": [1, 15, 16, 17], "target": [1, 5, 12, 14, 15, 17], "numer": [1, 8, 9, 15, 17], "make": [1, 2, 3, 5, 8, 11, 12, 16], "x": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17], "y_encod": 1, "astyp": [1, 12, 14], "categori": [1, 17], "cat": 1, "encod": [1, 15], "y": [1, 2, 3, 5, 9, 11, 12, 14, 15, 16], "map": [1, 15, 16], "cmap_light": 1, "pink": 1, "lightblu": 1, "lightgreen": 1, "cmap_bold": 1, "green": 1, "red": [1, 6, 8], "blue": [1, 5, 6, 8], "want": [1, 6, 8, 17], "let": [1, 2, 3, 5, 6, 11, 12, 14, 15, 16, 17], "clf": 1, "fit": [1, 2, 3, 5, 8, 12, 14, 17], "boundari": 1, "For": [1, 3, 5, 12, 15, 16, 17], "assign": [1, 16], "each": [1, 2, 3, 5, 9, 11, 17], "point": [1, 3, 5, 9, 16], "mesh": 1, "x_min": [1, 9], "x_max": [1, 9], "0": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "min": [1, 17], "max": [1, 17], "y_min": [1, 9], "y_max": [1, 9], "h": [1, 5], "02": [1, 8, 12, 14, 15, 16], "step": [1, 2, 5], "xx": 1, "yy": 1, "meshgrid": [1, 2], "arang": 1, "z": [1, 19], "c_": 1, "ravel": 1, "put": [1, 8, 9], "result": [1, 8, 11, 15], "reshap": [1, 8], "shape": [1, 5], "contourf": 1, "cmap": 1, "observ": [1, 5, 9, 11, 15, 16], "scatterplot": 1, "hue": 1, "palett": 1, "alpha": [1, 8, 9], "edgecolor": 1, "black": 1, "f": [1, 2, 3, 5, 6, 8, 9, 16, 17], "recal": [1, 2, 3, 5], "chapter": [1, 5, 9, 12, 15, 16, 17], "have": [1, 3, 5, 11, 12, 14, 15, 16, 17], "tabl": [1, 3, 16], "ml": [1, 2, 5, 9, 15, 16, 17], "contain": [1, 15, 16, 17, 19], "20": [1, 2, 5, 6, 9, 15], "ar": [1, 8, 9, 11, 12, 14, 15, 16, 17], "mix": [1, 16], "measur": [1, 17], "record": 1, "them": [1, 9], "first": [1, 3, 5, 16], "two": [1, 2, 5, 12, 14, 15, 16], "column": [1, 12, 14, 15, 17], "its": [1, 2, 3, 5, 6, 7, 12, 16, 17], "type": [1, 17], "class": [1, 8, 14, 15], "repres": [1, 5, 11, 16, 17], "last": 1, "91": [1, 15], "76": 1, "7": [1, 5, 8, 12, 14, 15, 16, 17], "09": [1, 15, 16], "69": [1, 3], "48": 1, "32": [1, 6, 8, 17], "9": [1, 6, 9, 12, 14, 15, 17], "21": [1, 2, 3, 5, 6, 12, 14, 15], "95": [1, 2, 3, 5, 6, 8, 15], "90": [1, 6, 8, 15], "62": [1, 6], "51": 1, "8": [1, 5, 6, 8, 12, 14, 15], "42": 1, "6": [1, 6, 8, 11, 12, 14, 15, 16], "19": [1, 6, 8, 12, 14, 15, 16], "50": [1, 2, 6, 8, 14, 15], "99": [1, 12, 14, 15], "15": [1, 5, 12, 15], "60": [1, 2, 3, 5, 6, 8], "29": [1, 6, 12, 15, 16, 17], "38": 1, "49": [1, 2, 3, 5, 6, 8, 14], "52": [1, 14], "44": [1, 17], "89": 1, "86": 1, "93": [1, 15], "12": [1, 5, 6, 12, 14, 15], "40": [1, 6, 8, 12, 14, 15], "82": [1, 3, 5], "re": [1, 2, 3], "aforement": 1, "manag": 1, "pan": 1, "el": 1, "da": 1, "ta": 1, "et": 1, "so": [1, 3, 5, 9, 11, 15], "note": [1, 8, 9, 11], "alreadi": [1, 3, 5], "begin": [1, 3, 5, 9, 17], "data": [1, 5, 18, 19], "transform": [1, 18], "datafram": [1, 9, 15, 16, 17], "print": [1, 2, 3, 5, 6, 8, 11, 12, 14, 15, 16, 17], "output": [1, 12], "13": [1, 2, 12, 14, 15], "14": [1, 15, 16], "16": [1, 3, 8, 15], "17": [1, 5, 6, 8, 12, 15], "18": [1, 2, 3, 5, 8, 12, 14, 15], "same": [1, 3, 5, 9, 11, 16], "had": [1, 5], "To": [1, 5, 9, 16, 17], "thing": [1, 3, 17], "easier": 1, "simpli": [1, 3, 5, 16], "call": [1, 3, 5, 8, 11, 17], "funtion": 1, "some": [1, 12, 16, 17], "firstli": [1, 11], "second": [1, 5, 16], "algorithm": [1, 3, 5, 8, 11, 16, 17, 19], "load": [1, 5, 14], "final": [1, 8], "pass": [1, 5], "feed": 1, "In": [1, 2, 3, 7, 9, 14, 16], "environ": [1, 14], "pleas": [1, 14], "rerun": [1, 14], "cell": [1, 14], "html": [1, 14, 16], "trust": [1, 14], "On": [1, 14], "github": [1, 12, 14], "unabl": [1, 14], "render": [1, 14], "try": [1, 9, 12, 14, 15, 16], "page": [1, 14], "nbviewer": [1, 14], "org": [1, 14, 16], "kneighborsclassifierkneighborsclassifi": 1, "readi": 1, "like": [1, 2, 3, 5, 9, 11, 17], "did": [1, 5, 8], "remov": [1, 16, 17], "datapoint": [1, 16], "after": [1, 5, 16, 17], "done": [1, 3, 5, 14], "anymor": 1, "henc": [1, 5], "paramet": [1, 2, 3, 11, 12, 17], "fals": [1, 14, 17], "see": [1, 2, 3, 5, 6, 9, 11, 12, 16], "figur": [1, 2, 5, 6, 9], "abov": [1, 3, 14, 16, 17], "specif": [1, 5], "new": [1, 3, 15, 16, 17], "instanc": [1, 3, 16], "an": [1, 2, 5, 7, 8, 9, 16, 17, 19], "4cm": 1, "5cm": 1, "land": 1, "9cm": 1, "3cm": 1, "singl": [1, 2, 9, 11, 17], "arrai": [1, 2, 3, 5, 6, 8, 11, 12, 14, 16], "dtype": [1, 5, 12, 14, 15, 16, 17], "object": [1, 5, 8, 12, 14, 16, 17], "whole": 1, "bunch": 1, "multipl": 1, "The": [1, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "next": [1, 2, 3, 5], "score": [1, 14], "rememb": 1, "explain": [1, 14], "should": [1, 12], "exampl": [1, 9, 16, 17], "pretend": [1, 2], "guess": [1, 16], "correctli": 1, "y_pred_train": 1, "actual": [1, 5], "calcul": [1, 2, 3, 9, 15, 16, 17], "between": [1, 2, 5, 15], "train_scor": 1, "accuracy_scor": 1, "mean": [1, 9, 16, 17], "100": [1, 2, 3, 5, 6, 9, 12, 14, 16], "train_error": 1, "imagin": [1, 15], "fuit": 1, "follow": [1, 2, 3, 5, 9, 11, 12, 14, 15], "properti": 1, "47": [1, 3, 12, 14], "01": [1, 8, 12, 14, 15], "34": [1, 15, 17], "23": [1, 3, 15, 16], "would": [1, 3, 5, 11, 16], "again": [1, 3], "reveal": 1, "compos": 1, "test_data": 1, "just": [1, 3, 5], "eyebal": 1, "made": 1, "wa": [1, 3, 7, 17], "verifi": 1, "rang": [1, 2, 3, 5, 9, 15, 16, 17], "len": [1, 2, 3, 5, 9], "pred": 1, "iloc": 1, "correct": 1, "els": 1, "wrong": 1, "incorrectli": [1, 16], "out": [1, 12, 14, 15], "As": [1, 3, 17], "when": [1, 2, 3, 5, 8, 12, 15, 16], "low": 1, "much": [1, 9, 16, 17], "higher": [1, 9], "unseen": [1, 11], "overfit": 1, "anoth": [1, 5, 15, 16], "problem": [1, 9, 15, 16, 17], "underfit": 1, "high": 1, "balanc": [1, 15], "pick": 1, "too": [1, 16], "simpl": [1, 2, 3, 5], "complic": 1, "happen": [1, 3], "select": [1, 11], "check": [1, 5, 8, 11, 15, 16], "onc": [1, 3, 11], "both": [1, 16], "onli": [1, 14, 15], "one": [1, 3, 8, 12, 15, 16], "dataapoint": 1, "ha": [1, 11, 12, 14, 15, 16, 17], "test_error": [1, 9], "test_misclassif": 1, "train_misclassif": 1, "_k": 1, "y_pred_test": 1, "compar": [1, 5, 9], "respons": 1, "y_train": [1, 6, 8, 9, 12, 14], "y_test": [1, 6, 8, 9, 12, 14], "test_scor": 1, "_train_error": 1, "round": [1, 2, 3, 5, 17], "_test_error": 1, "append": [1, 2, 3, 5, 12], "05": [1, 3, 6, 8, 12, 14, 15, 16], "35": [1, 15, 17], "45": [1, 15, 17], "55": [1, 6, 8, 14, 15], "user": [1, 5, 8, 14, 17], "andrewwolf": [1, 5, 8, 14, 17], "pyenv": [1, 5, 8, 14, 17], "version": [1, 5, 8, 14, 17], "lib": [1, 5, 8, 14, 17], "python3": [1, 5, 8, 14, 17], "site": [1, 5, 8, 14, 17], "packag": [1, 5, 8, 9, 14, 17], "base": [1, 16, 17], "py": [1, 5, 8, 14, 16, 17], "402": 1, "userwarn": [1, 8, 14], "featur": [1, 8, 15, 16, 18], "name": [1, 9, 12, 15, 17], "without": [1, 16], "warn": [1, 9, 14, 17], "vari": 1, "number": [1, 2, 3, 5, 8, 9, 11, 15, 16, 17], "axi": [1, 2, 5, 9, 17], "neighbour": 1, "perform": [1, 9, 11, 12, 14, 16, 17], "vs": 1, "left": [1, 16], "side": 1, "larg": 1, "neighborhood": 1, "produc": [1, 2], "small": [1, 3], "optim": 1, "other": [1, 3, 9, 11, 16, 17], "word": [1, 3], "thei": [1, 9, 16], "learn": [2, 3, 6, 8, 9, 11, 12, 14, 15, 16, 17, 19], "how": [2, 3, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17], "previou": [2, 3, 7], "section": [2, 3, 5, 16], "explor": [2, 3], "linear": [2, 4, 7, 8, 14], "regress": [2, 3, 4, 6, 7, 14], "onto": 2, "your": [2, 5, 9, 16, 17], "dataset": [2, 3, 6, 11, 15, 16, 17], "well": [2, 8, 9, 11], "quantifi": [2, 3], "good": [2, 3], "further": [2, 3], "our": [2, 3, 9, 16, 17], "hypothet": [2, 3, 16], "six": [2, 3, 5, 16], "apart": [2, 3, 5, 6], "locat": [2, 3, 5, 9, 15], "center": [2, 3, 5], "amsterdam": [2, 3, 5, 15, 17], "along": [2, 3, 5, 9], "price": [2, 3, 5, 6, 12, 14], "000": [2, 3, 5, 6, 12, 17], "eur": [2, 3, 5, 17], "floor": [2, 3, 5], "area": [2, 3, 5, 6], "squar": [2, 3, 6, 9, 14], "meter": [2, 3, 5, 6], "m": [2, 3, 5, 16], "30": [2, 3, 5, 6, 8, 11, 12, 14, 15, 17], "31": [2, 3, 5, 6, 8, 14, 15], "46": [2, 3, 5, 6, 8, 17], "80": [2, 3, 5, 6, 8], "65": [2, 3, 5, 6, 8], "77": [2, 3, 5, 6, 8], "70": [2, 3, 5, 6, 8, 11], "118": [2, 3, 5, 6, 8], "prettier": [2, 3, 5], "plot": [2, 3, 5, 6, 8], "creat": [2, 3, 8, 12, 15, 16, 17], "ve": [2, 3, 5], "been": [2, 3, 12, 16], "train": [2, 3, 5, 6, 8, 9, 11], "obtain": [2, 3, 15], "3x": [2, 5], "unknown": [2, 16], "leav": [2, 3, 17], "hat": [2, 5, 9], "random": [2, 3, 5, 9, 14], "valu": [2, 3, 5, 8, 9, 12, 15, 17], "evalu": [2, 5, 9, 11], "4": [2, 8, 14, 15], "a_rang": 2, "linspac": [2, 6, 8, 9, 17], "5": [2, 6, 8, 11, 12, 14, 15, 17], "differ": [2, 5, 15, 16, 17], "n": [2, 6, 9, 12, 14], "45833333": 2, "91666667": 2, "375": 2, "16666667": 2, "70833333": 2, "25": [2, 3, 5, 9, 15, 17], "79166667": 2, "33333333": [2, 17], "875": 2, "3": [2, 15], "41666667": 2, "95833333": 2, "ssr_rang": 2, "y_pred": [2, 3, 5, 9], "r_rang": 2, "predict": [2, 3, 6, 8, 9, 16], "residu": [2, 3], "save": 2, "list": [2, 3, 5, 6, 12, 14, 17], "sum": [2, 3, 6, 8, 9, 16], "282654": 2, "4583333333333335": 2, "197923": 2, "40798611112": 2, "9166666666666667": 2, "128329": 2, "46527777778": 2, "73872": 2, "171875": 2, "16666666666666652": 2, "34551": 2, "52777777779": 2, "708333333333333": 2, "10367": 2, "53298611112": 2, "1320": 2, "1875": 2, "7916666666666665": 2, "7409": 2, "491319444441": 2, "333333333333333": 2, "28635": 2, "444444444427": 2, "64998": 2, "046875": 2, "416666666666666": 2, "116497": 2, "29861111105": 2, "958333333333333": 2, "183133": 2, "19965277772": 2, "264905": 2, "75": 2, "nice": [2, 3, 5], "correspond": [2, 12, 15, 16], "loop": [2, 3, 5], "over": [2, 9, 15, 16], "chang": [2, 7, 16], "where": [2, 3, 5, 9, 11, 17], "shown": [2, 3, 9, 11], "6a": 2, "j": [2, 3, 17], "a0": 2, "a1": [2, 3], "return": [2, 6, 8, 9, 12, 14, 16], "d": [2, 3, 5, 9, 16], "coeffici": [2, 3, 5, 6], "c": [2, 5], "c0": 2, "1f": 2, "look": [2, 3, 12, 14, 16, 17], "intercept": [2, 5], "mpl_toolkit": 2, "mplot3d": 2, "axes3d": 2, "add_subplot": 2, "project": 2, "3d": 2, "aa0": 2, "aa1": 2, "plot_surfac": 2, "view_init": 2, "150": 2, "build": [3, 16], "remin": 3, "here": [3, 9, 14, 16, 17], "sympi": 3, "sym": 3, "But": [3, 16, 17], "end": [3, 9, 17], "procedur": [3, 9], "understood": 3, "ahead": 3, "format": [3, 6, 11, 12, 14, 17], "set": [3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "140": [3, 5], "110": [3, 5, 6, 8], "ok": [3, 9], "ident": 3, "visual": [3, 6, 8, 9], "represent": [3, 14], "ssr": 3, "more": [3, 8, 9, 16, 17], "overview": [3, 7], "prepar": [3, 5, 15, 16, 17], "empti": [3, 5], "41": [3, 5], "66": [3, 5, 6], "105": [3, 5, 12], "redisu": [3, 5], "139": [3, 5], "24": [3, 5, 12, 14, 15, 17], "400": [3, 5], "306": [3, 5], "146": [3, 5], "156": [3, 5], "1248": [3, 5], "1500000000003": 3, "equat": [3, 17], "split": [3, 11, 12, 14, 17], "big": [3, 17], "y_i": [3, 5], "ax_i": [3, 5], "respect": [3, 17], "ani": [3, 6, 16], "special": 3, "instal": [3, 12], "symbol": [3, 12, 14], "diff": 3, "displaystyl": 3, "51590": 3, "67218": 3, "find": [3, 5, 16], "lowest": 3, "becaus": [3, 5, 16, 17], "doe": [3, 8, 9], "minim": [3, 15], "which": [3, 5, 9, 11, 12, 14, 15, 16, 17], "plug": 3, "thu": 3, "slope": 3, "curv": [3, 9], "receiv": 3, "minimum": [3, 17], "determin": [3, 16], "multipli": 3, "rate": 3, "l": 3, "00001": 3, "step_siz": 3, "And": [3, 5], "updat": [3, 16], "a_": 3, "At": 3, "iter": [3, 8], "move": 3, "logic": 3, "32540": 3, "2338": 3, "3254": 3, "a2": 3, "99758": 3, "15752": 3, "7272": 3, "15753": 3, "a3": 3, "15511": 3, "7625": 3, "8952": 3, "07626": 3, "a4": 3, "23137": 3, "3691": 3, "6959": 3, "03692": 3, "a5": 3, "26829": 3, "1787": 3, "01787": 3, "a6": 3, "28616": 3, "865": 3, "1593": 3, "00865": 3, "a7": 3, "29481": 3, "418": 3, "8236": 3, "00419": 3, "a8": 3, "299": [3, 6], "202": 3, "7525": 3, "00203": 3, "a9": 3, "30102": 3, "98": [3, 15], "1525": 3, "00098": 3, "a10": 3, "30201": 3, "5156": 3, "00048": 3, "a11": 3, "30248": 3, "0023": 3, "00023": 3, "a12": 3, "30271": 3, "1354": 3, "00011": 3, "a13": 3, "30282": 3, "3907": 3, "5e": 3, "a14": 3, "30288": 3, "6096": 3, "3e": 3, "a15": 3, "3029": 3, "eventu": 3, "proper": 3, "succeed": 3, "47827": 3, "478277": 3, "75172": 3, "23153": 3, "3896": 3, "52019": 3, "11208": 3, "5559": 3, "11209": 3, "4081": 3, "5426": 3, "0619": 3, "05426": 3, "35384": 3, "2626": 3, "7566": 3, "02627": 3, "32758": 3, "1271": 3, "6129": 3, "01272": 3, "31486": 3, "615": 3, "5878": 3, "00616": 3, "3087": 3, "298": [3, 16], "006": 3, "00298": 3, "30572": 3, "144": 3, "2647": 3, "00144": 3, "30428": 3, "8386": 3, "0007": 3, "30358": 3, "33": [3, 6], "8088": 3, "00034": 3, "30324": 3, "3669": 3, "00016": 3, "30308": 3, "9232": 3, "8e": 3, "303": 3, "8356": 3, "4e": 3, "30296": 3, "8568": 3, "2e": 3, "30294": 3, "successfulli": 3, "even": [3, 16], "though": [3, 16], "cost": [4, 7], "function": [4, 7, 8, 11, 12], "gradient": [4, 7], "descent": [4, 7, 8], "linear_model": [5, 8, 14], "linearregress": [5, 8, 14], "length": 5, "model": [5, 7, 8, 9, 11, 14, 16, 17], "reg": 5, "estim": [5, 6, 8, 9, 14], "coef_": [5, 8], "intercept_": 5, "math": 5, "briefli": 5, "remind": 5, "better": 5, "wai": [5, 7, 12, 15, 16, 17], "core": 5, "shape_bas": 5, "visibledeprecationwarn": 5, "ndarrai": 5, "rag": 5, "nest": 5, "sequenc": 5, "tupl": 5, "deprec": 5, "meant": 5, "must": [5, 17], "specifi": [5, 11, 17], "ari": 5, "asanyarrai": 5, "vertic": 5, "line": [5, 9], "order": [5, 9, 16], "formula": 5, "_i": 5, "r_i": 5, "x_i": 5, "r_squar": 5, "10x": 5, "780": 5, "388806": 5, "388": 5, "806": 5, "proce": 5, "execut": 5, "task": [5, 17], "190": [5, 16], "4x": 5, "20326": 5, "326": 5, "might": [5, 8, 16], "notic": 5, "shrink": 5, "lower": [5, 9], "x_train": [6, 8, 9, 12, 14], "x_test": [6, 8, 9, 12, 14], "57": [6, 8], "85": [6, 8], "figsiz": [6, 9], "test": [6, 8, 11, 16], "loc": [6, 8, 9], "best": [6, 8, 12], "0x118f91c00": 6, "p": [6, 9, 12], "poly1d": [6, 9], "polyfit": [6, 9], "130": [6, 12, 14], "coef": [6, 8], "014425999538340081": 6, "4973416247674718": 6, "898294657797386": 6, "absolut": 6, "ab": [6, 12], "4100622821032": 6, "014": 6, "414": 6, "built": [6, 16], "given": 6, "614": 6, "961445499279304": 6, "altern": [6, 16], "predict_train": 6, "predict_test": 6, "994": 6, "7785614408572": 6, "1530": 6, "3762231241067": 6, "1027": 6, "0004120000003": 6, "1757": 6, "0769119999998": 6, "120": [6, 8], "489668977511541e": 6, "020758975169594147": 6, "8214724130889242": 6, "4626504642182": 6, "876": 6, "8597601245539": 6, "945": 6, "1647268737204": 6, "821": 6, "02076": 6, "0000849": 6, "249": 6, "579407116841026": 6, "651": 6, "4179373305931": 6, "29010": 6, "616059824526": 6, "57940712": 6, "33905077": 6, "72388224": 6, "67": 6, "03384222": 6, "96521691": 6, "35860073": 6, "4166544": 6, "61": 6, "044": 6, "0280625": 6, "63": 6, "0571009": 6, "113": 6, "7780625": 6, "688": 6, "6615471596378": 6, "29379": 6, "046097639017": 6, "0177085755377384e": 6, "00944944287510749": 6, "1443256656628589": 6, "75349695585578": 6, "1866": 6, "2074401186833": 6, "19915": 6, "12337120615": 6, "017709e": 6, "009449443": 6, "144326": 6, "7535": 6, "430313e": 6, "001865759": 6, "24949": 6, "27": [6, 12, 14, 15, 17], "9861": 6, "996": 6, "12053": 6, "21849": 6, "217305620088": 6, "238113566313": 6, "5344": 6, "177524015313": 6, "12329639": 6, "163138662402778e": 6, "6719065": 6, "318875373": 6, "6025432434314306": 6, "6718669": 6, "713593046": 6, "present": [7, 16], "focus": [7, 16], "modifi": 7, "complex": [7, 9, 17], "basi": 7, "expans": 7, "regular": 7, "automat": 8, "polynomi": 8, "preprocess": [8, 15, 17], "polynomialfeatur": 8, "lassocv": 8, "pipelin": 8, "make_pipelin": 8, "scale": 8, "confusingli": 8, "lambda": [8, 9], "term": 8, "via": 8, "argument": [8, 16], "default": [8, 16, 17], "full": 8, "penalti": 8, "degree_": 8, "lambda_": 8, "prevent": 8, "error": [8, 16], "polyx": 8, "degre": 8, "fit_transform": [8, 12, 15, 17], "model1": 8, "model2": 8, "ol": 8, "str": 8, "64": 8, "66185222664129": 8, "221838297484756": 8, "_ridg": 8, "216": 8, "linalgwarn": 8, "ill": 8, "condit": [8, 9], "matrix": 8, "rcond": 8, "10221e": 8, "mai": [8, 16], "accur": 8, "linalg": 8, "solv": 8, "A": [8, 11, 16, 19], "xy": 8, "assume_a": 8, "po": 8, "overwrite_a": 8, "t_": 8, "No": 8, "artist": 8, "whose": [8, 12], "underscor": 8, "ignor": [8, 9, 17], "max_it": 8, "1300000": 8, "var": [8, 9, 16], "folder": [8, 16], "5y": [8, 16], "7zvhsc3x5nx162713kvx9c1m0000gn": [8, 16], "ipykernel_8211": 8, "4278195487": 8, "With": [8, 12], "converg": 8, "advis": 8, "_coordinate_desc": 8, "634": 8, "coordin": 8, "lead": [8, 16, 17], "unexpect": 8, "discourag": 8, "cd_fast": 8, "enet_coordinate_desc": 8, "00000000e": [8, 12], "00": [8, 12, 14, 15], "64626506e": 8, "82147242e": 8, "07589752e": 8, "48966900e": 8, "77567467e": 8, "58872837e": 8, "81456434e": 8, "44216108e": 8, "convergencewarn": 8, "increas": 8, "consid": [8, 16, 17], "regularis": 8, "dualiti": 8, "gap": [8, 15], "281e": 8, "toler": 8, "672e": 8, "null": 8, "weight": [8, 16], "l1": 8, "effici": 8, "solver": 8, "ridgecv": 8, "instead": [8, 16, 17], "56": [8, 14, 15], "186089506337865": 8, "usual": [9, 16, 17], "known": 9, "sin": 9, "sigma_": 9, "epsilon": 9, "normal": [9, 17], "distribut": [9, 15], "nois": 9, "sim": 9, "probabl": 9, "uniform": 9, "interv": 9, "pi": 9, "case": [9, 16, 17], "frac": [9, 17], "text": 9, "tab": 9, "command": 9, "align": 9, "charact": 9, "otherwis": 9, "sigma_ep": 9, "irreduc": 9, "sample_x": 9, "sampl": [9, 11], "uniformli": 9, "sample_xi": 9, "draw": 9, "todo": 9, "sai": 9, "recov": 9, "seed": 9, "plt_new_fig": 9, "set_ax": 9, "limit": 9, "grid": 9, "plot_two_dataset": 9, "x1": 9, "y1": 9, "x2": 9, "y2": 9, "appropri": 9, "og": 9, "than": [9, 14, 16, 17], "help": [9, 17], "affect": [9, 12, 17], "200": 9, "integr": 9, "degrees_to_displai": 9, "filterwarn": 9, "ingor": 9, "poorli": 9, "fit_and_evaluate_polynomi": 9, "poli": 9, "ploc": 9, "clip": 9, "stabil": 9, "fit_and_plot_two_dataset": 9, "fhat_1": 9, "fhat_2": 9, "random_fit": 9, "num_fit": 9, "gener": [9, 11, 12], "zero": 9, "compute_e_fhat": 9, "20000": 9, "approxim": 9, "averag": [9, 16], "e": 9, "compute_vari": 9, "variance_x": 9, "variance_tot": 9, "e_fhat": 9, "plot_fits_efhat": 9, "smaller": [9, 16], "speed": 9, "up": 9, "orang": 9, "linewidth": 9, "4f": 9, "below": [9, 14, 16], "expect": 9, "e_": 9, "mathcal": 9, "against": 9, "e_x": 9, "int_": 9, "approx": [9, 17], "sum_": 9, "space": 9, "discret": [9, 17], "x_1": 9, "ldot": 9, "x_t": 9, "f_eval": 9, "compute_bias_sq": 9, "bias_sq_x": 9, "bias_sq": 9, "t_integral_area": 9, "plot_f_and_e_fhat": 9, "expeect": 9, "easi": 9, "displai": [9, 15], "df_epe": 9, "index": [9, 12, 14, 16, 17], "linestyl": 9, "marker": 9, "497500": 9, "027638": 9, "0625": 9, "587638": 9, "200318": 9, "031793": 9, "294611": 9, "004993": 9, "023503": 9, "090996": 9, "000483": 9, "454109": 9, "517092": 9, "previous": 9, "theoret": 9, "kei": [9, 16], "question": [9, 16], "match": 9, "emper": 9, "standard": [9, 16], "fit_and_eval_test_error": 9, "mean_sq_err": 9, "seri": [9, 12, 15, 16], "to_fram": 9, "agre": 9, "reasonbl": 9, "concat": [9, 17], "807306": 9, "340867": 9, "062224": 9, "156872": 9, "howev": [9, 17], "obvious": 9, "influenc": [9, 17], "natur": 9, "emperirc": 9, "mani": [9, 11, 16, 17], "emperir": 9, "veri": [9, 15, 17], "draw_sample_fit_and_eval_test_error_1": 9, "tranin": 9, "draw_samples_fit_and_eval_test_error": 9, "num_trial": 9, "10000": 9, "_": 9, "589134": 9, "291031": 9, "090609": 9, "426393": 9, "comptu": 9, "ideal": 9, "bia": 10, "varianc": 10, "decomposit": 10, "valid": [10, 14], "method": [10, 13, 16, 17, 19], "randomli": 11, "entir": 11, "train_test_split": [11, 12, 14], "test_siz": [11, 12, 14], "control": 11, "percentag": 11, "alloc": 11, "remain": [11, 14], "model_select": [11, 12, 14], "ratio": 11, "chosen": 11, "roughli": 11, "resampl": 11, "group": [11, 17], "time": [11, 15, 16], "kfold": 11, "n_split": [11, 14], "combin": 11, "appli": [11, 12, 15, 16], "leavepout": 11, "item": 11, "leaveoneout": 11, "dimension": 12, "reduct": 12, "techniqu": [12, 15, 19], "everyon": [12, 15, 16, 17], "read_csv": [12, 14], "http": [12, 14, 16], "5x12": [12, 14], "raw": [12, 14], "master": [12, 14], "car_pric": [12, 14], "csv": [12, 14], "delimit": [12, 14], "header": [12, 14], "head": [12, 14, 15], "car_id": [12, 14], "carnam": [12, 14], "fueltyp": [12, 14], "aspir": [12, 14], "doornumb": [12, 14], "carbodi": [12, 14], "drivewheel": [12, 14], "engineloc": [12, 14], "wheelbas": [12, 14], "engines": [12, 14], "fuelsystem": [12, 14], "boreratio": [12, 14], "stroke": [12, 14], "compressionratio": [12, 14], "horsepow": [12, 14], "peakrpm": [12, 14], "citympg": [12, 14], "highwaympg": [12, 14], "alfa": [12, 14], "romero": [12, 14], "giulia": [12, 14], "ga": [12, 14], "std": [12, 14, 16], "convert": [12, 14, 15, 16, 17], "rwd": [12, 14], "front": [12, 14], "88": [12, 14], "mpfi": [12, 14], "68": [12, 14], "111": [12, 14], "5000": [12, 14], "13495": [12, 14], "stelvio": [12, 14], "16500": [12, 14], "quadrifoglio": [12, 14], "hatchback": [12, 14], "94": [12, 14, 15], "152": [12, 14], "154": [12, 14], "26": [12, 14, 15], "audi": [12, 14], "ls": [12, 14], "sedan": [12, 14], "fwd": [12, 14], "109": [12, 14], "102": [12, 14], "5500": [12, 14], "13950": [12, 14], "100l": [12, 14], "4wd": [12, 14], "136": [12, 14], "115": [12, 14], "22": [12, 14, 15], "17450": [12, 14], "row": [12, 14, 16], "carlength": [12, 14], "carwidth": [12, 14], "carheight": [12, 14], "curbweight": [12, 14], "enginetyp": [12, 14], "cylindernumb": [12, 14], "random_st": [12, 14], "examin": [12, 14], "feature_select": [12, 14], "chi2": 12, "08315044e": 12, "11205757e": 12, "00159576e": 12, "66003574e": 12, "42430375e": 12, "04": [12, 14, 15], "87890909e": 12, "03": [12, 14, 15, 16], "04460495e": 12, "27081156e": 12, "02528346e": 12, "31340296e": 12, "77758862e": 12, "34366122e": 12, "09407540e": 12, "33440717e": 12, "001": 12, "20242844e": 12, "304": 12, "51419631e": 12, "004": 12, "47290251e": 12, "007": 12, "24387135e": 12, "005": 12, "chi_featur": 12, "sort_valu": 12, "ascend": 12, "inplac": [12, 16], "sort": 12, "000000e": 12, "202428e": 12, "472903e": 12, "07": [12, 15, 16], "243871e": 12, "514196e": 12, "334407e": 12, "float64": 12, "feature_nam": 12, "feature_scor": 12, "zip": [12, 14, 16], "2024284431006599e": 12, "00015141963086236825": 12, "4729025138749586e": 12, "243871349461334e": 12, "skfeatur": 12, "similarity_bas": 12, "fisher_scor": 12, "f_valu": 12, "int64": [12, 15], "pip": 12, "relieff": 12, "fs": 12, "n_features_to_keep": 12, "rank": 12, "relief_valu": 12, "origin": [12, 15, 16, 17], "suggest": 12, "yield": 12, "neg": [12, 16], "confid": 12, "impli": 12, "redund": 12, "commonsens": 12, "knowledg": 12, "strongest": 12, "car": [12, 17], "That": 12, "why": 12, "care": 12, "sever": [12, 16], "pattern": 12, "top": [12, 14], "relief_featur": 12, "satisfi": 12, "criteria": 12, "cor": 12, "corr": 12, "cor_target": 12, "relevant_featur": 12, "835305": 12, "874145": 12, "808138": 12, "filter": 13, "search": 13, "mlxtend": 14, "sequentialfeatureselector": 14, "sf": 14, "ensembl": 14, "randomforestclassifi": 14, "forest": 14, "sff": 14, "n_estim": 14, "k_featur": 14, "sbf": 14, "float": 14, "verbos": 14, "accuraci": 14, "cv": 14, "cross": 14, "parallel": 14, "n_job": 14, "backend": 14, "sequentialbackend": 14, "concurr": 14, "worker": 14, "_split": 14, "700": 14, "least": 14, "popul": 14, "member": 14, "less": 14, "elaps": 14, "1s": 14, "0s": 14, "5s": 14, "finish": 14, "2024": 14, "08": [14, 15, 16], "028071205007824725": 14, "4s": 14, "03501564945226917": 14, "3s": 14, "04196009389671361": 14, "2s": 14, "53": 14, "k_feature_idx_": 14, "top_forward": 14, "bottom": 14, "r2": 14, "sfs1": 14, "8s": 14, "766709228853002": 14, "6s": 14, "784601658645056": 14, "58": [14, 15], "7791013015371373": 14, "59": 14, "7709674959750785": 14, "7774843727364775": 14, "7830498884361807": 14, "9s": 14, "8152003870787854": 14, "7s": 14, "7464561632796359": 14, "7565910493715275": 14, "top_backward": 14, "rfe": 14, "rfem": 14, "n_features_to_select": 14, "rferf": 14, "linearregressionlinearregress": 14, "top_recurs": 14, "support_": 14, "dict": 14, "ranking_": 14, "imblearn": 15, "over_sampl": 15, "smote": 15, "labelencod": [15, 17], "silver": 15, "Suchs": 15, "bank": 15, "transact": 15, "certain": 15, "period": 15, "statu": 15, "fraud": 15, "fraudul": 15, "legit": 15, "legal": 15, "These": 15, "imbalanc": 15, "date": [15, 16], "2020": [15, 16], "06": 15, "28": 15, "dusseldorf": 15, "berlin": 15, "belgium": 15, "pari": 15, "df_bank_transact": 15, "2": 15, "status_count": 15, "value_count": 15, "count": 15, "classif": 15, "minor": 15, "major": 15, "signific": 15, "lot": 15, "One": [15, 16], "oversampl": 15, "demonstr": [15, 17], "le_loc": 15, "location_encod": 15, "le_statu": 15, "status_encod": 15, "sampling_strategi": 15, "k_neighbor": 15, "x_re": 15, "y_re": 15, "fit_resampl": 15, "option": 15, "back": [15, 17], "resampled_data": 15, "inverse_transform": 15, "tail": 15, "92": 15, "96": 15, "97": 15, "dictionari": [16, 17], "custom": 16, "id": 16, "383": 16, "1997": 16, "698": 16, "1314": 16, "333": 16, "1996": 16, "state": 16, "pennsylvania": 16, "californai": 16, "california": 16, "iowa": 16, "york": 16, "washington": 16, "drexel": 16, "hill": 16, "sacramento": 16, "lo": 16, "angel": 16, "fort": 16, "dodg": 16, "brooklyn": 16, "postal": 16, "19026": 16, "94229": 16, "90058": 16, "50501": 16, "11249": 16, "98101": 16, "ship": 16, "243": 16, "193": 16, "consol": 16, "nan": 16, "dirti": 16, "onlin": 16, "product": 16, "issu": 16, "irrelev": 16, "imposs": 16, "awai": 16, "clearli": 16, "spell": 16, "misspel": 16, "mistak": 16, "were": [16, 19], "uncorrect": 16, "treat": 16, "string": [16, 17], "identifi": 16, "whether": [16, 17], "particular": 16, "uniqu": 16, "often": [16, 17], "itself": 16, "fix": 16, "replac": 16, "revisit": 16, "includ": [16, 17], "misalign": 16, "sinc": [16, 17], "inconsist": 16, "to_datetim": 16, "handl": 16, "mixtur": 16, "dynam": 16, "particularli": 16, "consist": 16, "datetim": 16, "iso": 16, "dd": 16, "mm": 16, "yyyi": 16, "convers": 16, "dt": 16, "strftime": 16, "common": [16, 17], "aris": 16, "practic": 16, "provid": [16, 17], "three": [16, 17], "four": 16, "five": 16, "effect": 16, "doubl": 16, "give": 16, "priorit": 16, "poor": 16, "drop_dupl": 16, "df_clean": 16, "varieti": 16, "reason": 16, "enter": 16, "human": 16, "being": 16, "he": 16, "forgotten": 16, "input": 16, "genuin": 16, "unmeasur": 16, "survei": 16, "answer": 16, "occur": 16, "run": [16, 17], "cannot": 16, "exactli": 16, "educ": 16, "imput": 16, "median": 16, "total": 16, "exclud": 16, "median_purchas": 16, "fillna": 16, "ipykernel_8292": 16, "3282763330": 16, "futurewarn": [16, 17], "copi": 16, "through": 16, "chain": 16, "behavior": 16, "never": 16, "intermedi": 16, "alwai": [16, 17], "behav": 16, "col": 16, "oper": 16, "settingwithcopywarn": 16, "slice": 16, "caveat": 16, "document": 16, "pydata": 16, "doc": 16, "stabl": 16, "user_guid": 16, "view": 16, "versu": 16, "sometim": 16, "fill": 16, "merg": 16, "main": 16, "accordingli": 16, "relat": 16, "zip_to_c": 16, "seattl": 16, "df_map": 16, "df_merg": 16, "suffix": 16, "_map": 16, "city_map": 16, "significantli": [16, 17], "problemat": 16, "sensit": 16, "heavili": 16, "consequ": 16, "rest": 16, "There": 16, "hard": 16, "fast": 16, "rule": 16, "about": 16, "subsect": 16, "statist": [16, 19], "commonli": 16, "simplest": 16, "far": 16, "mean_valu": 16, "std_dev": 16, "deviat": 16, "186": 16, "33333333333334": 16, "13124720419029": 16, "suppos": 16, "accept": 16, "Then": 16, "threshold": 16, "lower_bound": 16, "upper_bound": 16, "anyth": 16, "114": 16, "06040827923752": 16, "486": 16, "7270749459042": 16, "72": 16, "onehotencod": 17, "standardscal": 17, "ag": 17, "36": 17, "54": 17, "incom": 17, "95000": 17, "210000": 17, "75000": 17, "30000": 17, "55000": 17, "430000": 17, "truck": 17, "ye": 17, "downtown": 17, "suburb": 17, "clean": [17, 18], "consum": 17, "illustr": 17, "independ": 17, "person": 17, "live": 17, "oh": 17, "spars": 17, "encoded_featur": 17, "get_feature_names_out": 17, "concaten": 17, "df_encod": 17, "drop": 17, "_encod": 17, "808": 17, "renam": 17, "sparse_output": 17, "unless": 17, "vehicle_car": 17, "vehicle_non": 17, "vehicle_truck": 17, "kids_no": 17, "kids_y": 17, "le": 17, "year": 17, "430": 17, "unscal": 17, "technic": 17, "prohibit": 17, "larger": 17, "decis": 17, "tree": 17, "invari": 17, "necessarili": 17, "ensur": 17, "doesn": 17, "advers": 17, "scaler": 17, "columns_to_scal": 17, "498342": 17, "392844": 17, "414214": 17, "447214": 17, "897015": 17, "441194": 17, "707107": 17, "196020": 17, "537894": 17, "236068": 17, "099668": 17, "864257": 17, "797347": 17, "682945": 17, "694362": 17, "036746": 17, "process": 17, "either": 17, "continu": 17, "real": 17, "chop": 17, "young": 17, "qquad": 17, "middl": 17, "old": 17, "straightforward": 17, "cut": 17, "extend": 17, "possibl": 17, "inclus": 17, "allow": 17, "clear": 17, "meaning": 17, "analysi": 17, "factor": 17, "divid": 17, "hyper": 17, "comput": 17, "w": 17, "th": 17, "maximum": 17, "cdot": 17, "eq": 17, "equal_width_bin": 17, "demograph": 17, "integ": 17, "min_ag": 17, "max_ag": 17, "num": 17, "66666667": 17, "include_lowest": 17, "bin": 18, "augment": 18, "repositori": 19, "andrew": 19, "wolf": 19, "guid": 19, "world": 19, "scienc": 19, "describ": 19}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"chapter": [0, 4, 7, 10, 13, 18], "2": [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 16, 17], "k": [1, 11], "nearest": 1, "neighbor": 1, "classifi": 1, "1": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "requir": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "librari": [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17], "function": [1, 2, 3, 5, 9], "probelm": 1, "represent": [1, 5], "creat": [1, 5], "hypothet": [1, 5], "dataset": [1, 5, 9], "visual": [1, 5], "3": [1, 3, 4, 5, 6, 8, 9, 11, 12, 14, 16, 17], "learn": [1, 5], "predict": [1, 5], "build": [1, 2, 5, 6], "knn": 1, "vizual": [1, 5], "decis": 1, "region": 1, "unknown": 1, "valu": [1, 16], "4": [1, 3, 5, 6, 7, 9, 11, 12, 16, 17], "how": [1, 5], "good": [1, 5], "our": [1, 5], "evalu": 1, "train": 1, "set": 1, "test": [1, 9], "5": [1, 3, 5, 9, 10, 16], "model": [1, 6], "complex": 1, "view": 1, "differ": [1, 3, 9], "scenario": 1, "n": 1, "tune": 1, "hyperparamet": 1, "plot": [1, 9], "misclassif": 1, "error": [1, 9], "rate": 1, "cost": [2, 3], "data": [2, 3, 6, 8, 9, 11, 12, 14, 15, 16, 17], "ssr": [2, 5], "gradient": 3, "descent": 3, "action": 3, "deriv": 3, "step": [3, 14], "6": [3, 9, 13], "7": [3, 9, 18], "8": 3, "9": 3, "10": 3, "11": 3, "12": 3, "13": 3, "14": 3, "15": 3, "initi": 3, "linear": 5, "regress": [5, 8], "problem": 5, "draw": 5, "residu": 5, "calcul": [5, 6], "sum": 5, "squar": [5, 12], "wrong": 5, "paramet": 5, "i": 5, "ii": 5, "basi": 6, "expans": 6, "three": 6, "polynomi": [6, 9], "first": 6, "degre": [6, 9], "second": 6, "ssr_train": 6, "ssr_test": 6, "fourth": 6, "fifth": 6, "regular": 8, "ridg": 8, "lasso": 8, "bia": 9, "varianc": 9, "decomposit": 9, "synthet": 9, "defin": 9, "target": 9, "two": 9, "fit": 9, "comput": 9, "comptuat": 9, "total": 9, "ep": 9, "empir": 9, "conclus": 9, "cross": 11, "valid": 11, "method": [11, 12, 14], "variabl": [11, 12, 14, 15, 16, 17], "hold": 11, "out": 11, "fold": 11, "kfcv": 11, "leav": 11, "p": 11, "lpocv": 11, "One": [11, 17], "loocv": 11, "filter": 12, "chi": 12, "fisher": 12, "score": 12, "relief": 12, "correl": 12, "base": 12, "featur": [12, 14, 17], "select": [12, 14], "compar": [12, 14], "four": [12, 14], "search": 14, "wrapper": 14, "forward": 14, "backward": 14, "recurs": 14, "elimin": 14, "augment": 15, "clean": 16, "incorrect": 16, "improperli": 16, "format": 16, "duplic": 16, "miss": 16, "purchas": 16, "column": 16, "citi": 16, "outlier": 16, "transform": 17, "bin": 17, "encod": 17, "appli": 17, "hot": 17, "vehicl": 17, "kid": 17, "label": 17, "resid": 17, "scale": 17, "gener": 17, "approach": 17, "equal": 17, "width": 17, "about": 19}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx": 56}})