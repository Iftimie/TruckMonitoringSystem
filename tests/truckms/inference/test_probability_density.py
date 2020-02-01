from truckms.inference.neural import pred_iter_to_pandas
from truckms.api import PredictionDatapoint, model_class_names
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats.kde import gaussian_kde
import numpy as np

def test_estimate_next_frame_ids():
    frame_num_24_hours = 30 * 3600 * 24
    frame_num_24_hours = 30 * 3600 * 1

    def dummy_generator():
        for i in range(frame_num_24_hours):
            if 10 < i < 50 or 200 < i < 500 or 800 < i < 900 or 90000 < i < 95000:
                #simulate detection
                dp = PredictionDatapoint(pred={"boxes": np.array([[0, 0, 0, 0]]),
                                           "scores": np.array([0]),
                                           "labels": np.array([model_class_names.index("truck")]),
                                           "obj_id": np.array([0])},
                                     frame_id=i)
            else:
                #simulate no detection
                dp = PredictionDatapoint(pred={"boxes": np.array([[]]).reshape(-1, 4),
                                               "scores": np.array([]),
                                               "labels": np.array([]),
                                               "obj_id": np.array([])},
                                         frame_id=i)
            yield dp

    dummy_df = pred_iter_to_pandas(pdp_iterable=dummy_generator())
    cols = list(dummy_df.columns)
    cols.remove("reason")
    dummy_df = dummy_df.dropna(subset=cols)
    plt.hist(dummy_df['img_id'], range=(0, frame_num_24_hours))
    # plt.show()

    kde = gaussian_kde(dummy_df['img_id'])
    # these are the values over wich your kernel will be evaluated
    dist_space = np.linspace(0, frame_num_24_hours, 1000)
    # plot the results
    plt.plot(dist_space, kde(dist_space))
    # plt.show()


    import scipy.stats as st

    class TrafficPDF(st.rv_continuous):

        def __init__(self, dataframe):
            assert 'img_id' in dataframe.columns
            super(TrafficPDF, self).__init__(a=dataframe['img_id'].min() , b=dataframe['img_id'].max(), name="TrafficPDF")
            self.kde = gaussian_kde(dataframe['img_id'])

        # def _pdf(self, x):
        #     return self.kde(x)
        def _cdf(self, x):
            return self.kde.integrate_box_1d(-np.Inf, x)

    my_cv = TrafficPDF(dummy_df)

    generated_data = my_cv.rvs(size=100).astype(np.int32)
    generated_data = sorted(generated_data.tolist())
    plt.hist(generated_data, bins=100)
    # plt.show()

    detected_frames = set(dummy_df['img_id'].unique())

    frames_to_search = set()
    for i in generated_data:
        while i in frames_to_search or i in detected_frames:
            i += 1
        frames_to_search.add(i)


    assert len(set(detected_frames) & set(frames_to_search)) == 0
    pass