
def test_dataloader(learner, test_data, test_target=None):
  dls = learner.dls
  valid_dl = dls.valid
  if test_target is not None:
    # Labelled test data
    test_ds = valid_dl.dataset.add_test(test_data, test_target)# In this case I'll use X and y, but this would be your test data
    test_dl = valid_dl.new(test_ds)
  else:
    test_ds = dls.dataset.add_test(test_data)
    test_dl = valid_dl.new(test_ds)
  return test_dl