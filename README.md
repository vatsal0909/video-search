# 🎬 Video Search Application

A complete AI-powered video search application that lets you upload videos and search through them using natural language queries. Built with AWS services, React, and powered by Amazon Bedrock's Marengo models.

## ✨ Features

- 🎥 **Video Upload**: Upload videos directly through the web interface
- 🔍 **AI-Powered Search**: Search videos using natural language descriptions
- 🖼️ **Thumbnail Generation**: Automatic thumbnail creation for uploaded videos
- ⚡ **Real-time Processing**: Videos are processed automatically upon upload
- 🌐 **Global CDN**: Fast content delivery via CloudFront
- 🔒 **Secure**: Built with AWS security best practices

## 🚀 Quick Deploy (Fork & Deploy)

**Perfect for anyone who wants to try this application - no local setup required!**

### 1. Fork this repository
Click the "Fork" button at the top of this page (GitHub web interface)

### 2. Set up AWS credentials
In your forked repository, go to **Settings → Secrets and variables → Actions** and add:
- `AWS_ACCESS_KEY_ID` - Your AWS access key
- `AWS_SECRET_ACCESS_KEY` - Your AWS secret key

### 3. Deploy with one click
1. Go to the **Actions** tab in your forked repository
2. Click **Deploy Video Search Infrastructure**
3. Click **Run workflow**
4. Choose your settings (defaults work great for testing):
   - **Environment**: `demo`
   - **Stack prefix**: `vs-1` 
   - **AWS Region**: `us-east-1`
   - **Deploy frontend**: ✅ (checked)
   - **Action**: `deploy`
5. Click **Run workflow**

⏱️ **Deployment takes 15-25 minutes**. You'll get a complete working application!

🌐 **Everything happens in GitHub** - no local tools or setup required!

📖 **[Step-by-Step Visual Guide](GITHUB_WORKFLOW_GUIDE.md)** - Detailed screenshots and instructions

📖 **[Full Deployment Guide](DEPLOYMENT_GUIDE.md)** - Detailed instructions and troubleshooting

## 🏗️ Architecture

### Backend (AWS)
- **Amazon Bedrock**: AI video understanding with Marengo models
- **OpenSearch**: Vector search for video embeddings
- **ECS Fargate**: Scalable video processing
- **Lambda**: Serverless API functions
- **S3**: Video and asset storage
- **CloudFront**: Global content delivery

### Frontend (React)
- **React**: Modern UI framework
- **Tailwind CSS**: Utility-first styling
- **Vite**: Fast build tooling
- **S3 + CloudFront**: Static hosting

## 💻 Local Development (Optional)

**Note: Local development is completely optional!** You can deploy and use the entire application through GitHub's web interface.

### If you want to develop locally:
- Node.js 18+ (for frontend development)
- AWS CLI configured (for direct AWS access)
- No Docker required (uses pre-built images)

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

### Backend Development
Backend services run on AWS using pre-built Docker images. See individual service README files in the `backend/` directory for development details.

## 📊 Monitoring & Logs

After deployment, monitor your application:
- **CloudWatch Logs**: Application logs and errors
- **CloudFormation**: Infrastructure status
- **S3**: Uploaded videos and processed content
- **OpenSearch**: Search indices and performance


**Production**: Scales with usage. Set up billing alerts!

## 🧹 Cleanup

**Important**: To avoid ongoing charges, use the same workflow with cleanup action:
1. Go to **Actions** → **Deploy Video Search Infrastructure**
2. Set **Action** to `cleanup`
3. Use the same environment/prefix settings
4. Run the workflow

This will delete all AWS resources and stop billing.

## 🔧 Configuration

The application supports multiple environments:
- **demo**: Testing and demos
- **dev**: Development
- **stage**: Staging
- **prod**: Production

Each environment is isolated with its own resources.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📖 [Deployment Guide](DEPLOYMENT_GUIDE.md)
- 🐛 [Create an Issue](../../issues)
- 💬 [Discussions](../../discussions)

## 🎯 What's Next?

After deploying:
1. Upload test videos
2. Try different search queries
3. Explore the codebase
4. Customize for your needs
5. Build something amazing!

---

**Ready to get started?** 👆 Fork this repo and deploy in minutes!