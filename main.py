from data_reader import *
from features import *
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import  make_scorer
from sklearn.model_selection import cross_val_score


def compute_score(gt, predicted):
    """
    Root Mean Squared Logarithmic Error
    """
    return np.average(np.square(np.log(predicted + 1) - np.log(gt + 1)))**0.5


def evaluate(clf, features, target):
    predicted = clf.predict(features)
    return compute_score(target, predicted)


def test(forest):
    test_df = read_test(DATA_ROOT)
    test_features_filepath = os.path.join(DATA_ROOT, 'test_feats.hdf5')
    test_feats = extract_features(test_df,  test_features_filepath)
    print 'test features shape:', test_feats.values.shape
    pred_votes = forest.predict(test_feats.values)
    print 'Predicted votes shape:', pred_votes.shape

    result = pd.DataFrame(index=test_feats.index, data=pred_votes, columns=['Votes'])
    result_filepath = os.path.join(DATA_ROOT, 'result.csv')
    result.to_csv(os.path.join(DATA_ROOT, 'result.csv'), index_label='Id')
    print 'Saved to file', result_filepath


def create_forest(**args):
    return RandomForestRegressor(n_estimators=300, criterion='mse', max_depth=None, min_samples_split=5,
                                 min_samples_leaf=args.get('min_samples_leaf', 50),
                                 min_weight_fraction_leaf=0.0, max_features='auto',
                                 bootstrap=True, oob_score=True,
                                 n_jobs=2, random_state=None, verbose=True, warm_start=False)

if __name__ == '__main__':
    df, train_votes = read_train(DATA_ROOT)
    assert len(df) == len(train_votes)
    print 'Extracting train features'
    start = time.time()
    force_train_features = False
    train_features_filepath = os.path.join(DATA_ROOT, 'train_feats.hdf5')
    if os.path.exists(train_features_filepath) and not force_train_features:
        train_feats = pd.read_hdf(train_features_filepath, key='df')
    else:
        train_feats = extract_features(df, train_features_filepath)
    print 'Elapsed time on feature extraction: {} s'.format(time.time() - start)
    pass

    print train_feats.shape
    print train_votes.shape
    assert train_feats.shape[0] == train_votes.shape[0]

    print '======'
    print 'Cross validation'
    rmsle_scorer = make_scorer(compute_score, greater_is_better=False)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    clf = create_forest()
    scores = cross_val_score(clf, train_feats.values, train_votes.values, cv=kf, scoring=rmsle_scorer, n_jobs=2, verbose=True)
    print scores
    print("RMSLE: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    try:
        print 'R2 score:', clf.oob_score_
    except:
        print 'No R2 score'

    print '======'
    force_train = True
    should_dump_model = False
    forest_path = os.path.join(DATA_ROOT, 'forest.pkl')
    if not force_train and os.path.exists(forest_path):
        print 'Loading forest'
        with open(forest_path, 'r') as f:
            forest = pickle.load(f)
    else:
        full_forest = create_forest()

        print 'Training'
        full_forest.fit(train_feats.values, train_votes.values)
        if should_dump_model:
            with open(forest_path, 'wb') as f:
                pickle.dump(full_forest, f)

    print 'Predicting on train...'
    print 'Train RMSLE Score:', evaluate(full_forest, train_feats.values, train_votes.values)

    test(full_forest)


