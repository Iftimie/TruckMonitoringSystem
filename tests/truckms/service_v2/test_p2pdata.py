from truckms.service_v2.p2pdata import p2p_update, p2p_insert_one


def test_p2p_inser_one(tmpdir):
    db, col = "mydb", "movie_statuses"
    data = {"res": 320, "name": "somename"}

    def post_func(url, **kwargs):

    p2p_insert_one(db, col, )

def test_p2p_update(tmpdir):
    db = "mydb"
    col = "movie_statuses"

