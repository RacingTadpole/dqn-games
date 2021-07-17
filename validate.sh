# Set this as a git pre-commit hook with:
# cd .git/hooks && ln -s ../../validate.sh pre-commit && cd ../..
pylint games && mypy . && pytest
