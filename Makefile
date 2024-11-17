run:
	docker build -t fastapi-od .
	docker run -p 8000:8000 fastapi-od

