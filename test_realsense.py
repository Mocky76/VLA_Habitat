from scripts.iros_challenge.onsite_competition.sdk.save_obs import load_obs_from_meta
rs_meta_path = '/home/sjy/InternNav/scripts/iros_challenge/onsite_competition/captures/rs_meta.json'

fake_obs_640 = load_obs_from_meta(rs_meta_path)
fake_obs_640['instruction'] = 'go to the red car'
print(fake_obs_640['rgb'].shape, fake_obs_640['depth'].shape)