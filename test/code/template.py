def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.decay = '100'

    if args.template.find('EDSR_paper') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

    if args.template.find('MFRAN_x2') >= 0:
        args.model = 'MFRAN'
        args.scale = '2'
        args.patch_size = 96
        args.epochs = 1000
        args.FRBs = 16
        args.path = 4
        args.n_feats = 64

    if args.template.find('MFRAN_x3') >= 0:
        args.model = 'MFRAN'
        args.scale = '3'
        args.patch_size = 144
        args.epochs = 1000
        args.FRBs = 16
        args.path = 4
        args.n_feats = 64

    if args.template.find('MFRAN_x4') >= 0:
        args.model = 'MFRAN'
        args.scale = '4'
        args.patch_size = 192
        args.epochs = 1000
        args.FRBs = 16
        args.path = 4
        args.n_feats = 64

    if args.template.find('MFRAN_test') >= 0:
        args.model = 'MFRAN'
        args.scale = '2'
        args.FRBs = 16
        args.path = 4
        args.n_feats = 64
        args.self_ensemble = 'True'
        args.test_only = 'True'
        args.chop = 'True'