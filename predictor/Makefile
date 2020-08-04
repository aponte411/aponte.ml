.EXPORT_ALL_VARIABLES:
BUCKET_NAME=model_registry
VERSION=0.1

package_predictor:
	python setup.py sdist --formats=gztar
	gsutil cp dist/predictor-${VERSION}.tar.gz gs://${BUCKET_NAME}/inference/
