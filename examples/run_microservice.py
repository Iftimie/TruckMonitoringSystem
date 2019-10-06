import os.path as osp
from truckms.service.service import create_microservice

def main():
    up_dir = 'up_dir'
    app = create_microservice(up_dir, max_operating_res=800)
    app.run(host="0.0.0.0", port=5000)

if __name__ == '__main__':
    main()