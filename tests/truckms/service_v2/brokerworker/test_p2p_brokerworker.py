from truckms.service_v2.brokerworker.p2p_brokerworker import create_p2p_brokerworker_app

def test_create_p2p_brokerworker_app(tmpdir):
    app = create_p2p_brokerworker_app(tmpdir)
    assert app is not None