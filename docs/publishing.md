# Publish to GitHub

This is a local repository. No online repository, remote, or initial commit has been created automatically.

1. Review the scaffold, select a license and configure your own Git author identity.
2. On GitHub create an empty repository named beauty-video-templates; choose visibility appropriate to the project and do not initialize a README.
3. From this repository folder run:

```sh
git add .
git commit -m "chore: initialize research project scaffold"
git remote add origin https://github.com/YOUR_USERNAME/beauty-video-templates.git
git push -u origin main
```

Replace YOUR_USERNAME with your account. Use your normal authentication flow; never embed tokens in remote URLs or source files. Verify Actions after pushing and consider requiring CI before merging. Add the verified repository URL to package metadata later.
