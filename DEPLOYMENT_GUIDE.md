# 🚀 Video Search Deployment Guide

This guide helps you deploy the complete Video Search application using GitHub Actions workflows. Perfect for anyone who has forked this repository!

## 📋 Prerequisites

### 1. GitHub Account
- Free GitHub account
- Ability to fork repositories

### 2. AWS Account Setup
- Active AWS account with billing enabled
- AWS Access Key ID and Secret Access Key
- **No local AWS CLI needed** - everything runs in GitHub Actions
- Estimated cost: $10-50/month depending on usage

### 3. Required GitHub Secrets

Go to your **forked repository** → **Settings → Secrets and variables → Actions** and add:

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | AWS Access Key ID | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | AWS Secret Access Key | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |

### 4. AWS Permissions Required

Your AWS user/role needs these permissions:
- CloudFormation (full access)
- S3 (full access)
- Lambda (full access)
- ECS (full access)
- OpenSearch (full access)
- IAM (full access)
- VPC (full access)
- CloudFront (full access)

**Note: Everything runs in GitHub Actions - no local tools required!**

## 🎯 Quick Start (GitHub-Only Deployment)

**No local setup required - everything happens in your browser!**

### Step-by-Step Process:

1. **Fork this repository** (click Fork button above)
2. **Add AWS credentials** to your forked repo's secrets
3. **Go to Actions tab** in your forked repository
4. **Click "Deploy Video Search Infrastructure"**
5. **Click "Run workflow"**
6. **Configure and deploy** (see options below)
7. **Wait 15-25 minutes** for completion
8. **Get your application URL** from the workflow summary

### Option 1: Deploy Everything
1. Go to **Actions** tab in your GitHub repository
2. Click **Deploy Video Search Infrastructure**
3. Click **Run workflow**
4. Configure options:
   - **Environment**: `demo` (recommended for testing)
   - **Stack prefix**: `vs-1` (or your choice, max 8 chars)
   - **AWS Region**: `us-east-1` (recommended)
   - **Deploy frontend**: ✅ (checked)
   - **Action**: `deploy`
5. Click **Run workflow**

⏱️ **Total deployment time**: 15-25 minutes

### Option 2: Deploy Backend Only
Same as above, but uncheck **Deploy frontend** if you want to deploy frontend separately later.

## 📊 What Gets Deployed

### Backend Infrastructure
- **OpenSearch Domain**: For video embeddings and search
- **Lambda Functions**: Video processing and search APIs
- **ECS Fargate**: Video preprocessing and search services
- **Step Functions**: Video processing pipeline
- **S3 Buckets**: Video storage and processed content
- **VPC**: Secure networking with private subnets
- **CloudFront**: API distribution for global access
- **IAM Roles**: Least-privilege security

### Frontend Infrastructure
- **S3 Bucket**: Static website hosting
- **CloudFront**: Global CDN for fast loading
- **React App**: Built and deployed automatically

## 🔧 Configuration Options

### Environment Types
- **demo**: For testing and demos (lower costs)
- **dev**: Development environment
- **stage**: Staging environment  
- **prod**: Production environment

### Stack Prefix
- Must be 1-8 characters
- Alphanumeric and hyphens only
- Used to avoid naming conflicts
- Example: `vs-1`, `demo`, `test`

### AWS Regions
Supported regions:
- `us-east-1` (recommended - lowest latency for Bedrock)
- `us-east-2`
- `us-west-1`
- `us-west-2`
- `eu-west-1`
- `eu-central-1`

## 📱 Using Your Deployed Application

After successful deployment, you'll get:

### 🌐 Frontend URL
- Access your video search application
- Upload videos and perform searches
- Example: `https://d1234567890.cloudfront.net`

### 🔗 API URL  
- Backend API for integrations
- Health check endpoint: `/health`
- Example: `https://d0987654321.cloudfront.net`

### 🪣 Video Bucket
- S3 bucket for uploading videos
- Automatically triggers processing pipeline
- Example: `vs-1-videos-us-east-1-123456789012-demo`

## 🔍 Monitoring Your Deployment

### GitHub Actions
- Check the **Actions** tab for deployment progress
- View detailed logs for each step
- Get deployment summary with all URLs

### AWS Console
- **CloudFormation**: View stack resources and outputs
- **CloudWatch**: Monitor logs and metrics
- **S3**: Check uploaded videos and processed content
- **OpenSearch**: View search indices and data

## 🧹 Cleanup (Important!)

To avoid ongoing AWS charges:

1. Go to **Actions** tab
2. Click **Deploy Video Search Infrastructure**
3. Click **Run workflow**
4. Configure options:
   - **Environment**: Same as your deployment
   - **Stack prefix**: Same as your deployment
   - **AWS Region**: Same as your deployment
   - **Action**: `cleanup`
5. Click **Run workflow**

⚠️ **This will permanently delete all resources and data!**

## 🚨 Troubleshooting

### Common Issues

#### 1. "Stack already exists" error
- Use a different stack prefix
- Or delete the existing stack first

#### 2. "Insufficient permissions" error
- Check your AWS IAM permissions
- Ensure GitHub secrets are correct

#### 3. "OpenSearch domain creation failed"
- Try a different AWS region
- Check service limits in your AWS account

#### 4. Frontend not loading
- Wait 5-10 minutes for CloudFront propagation
- Check browser console for errors
- Verify the config.json file was uploaded

### Getting Help

1. **Check GitHub Actions logs** for detailed error messages
2. **AWS CloudFormation Events** show infrastructure deployment issues
3. **CloudWatch Logs** contain runtime errors
4. **Create an issue** in this repository with error details

## 💰 Cost Estimation

### Typical Monthly Costs (demo environment)
- **OpenSearch**: $15-25
- **ECS Fargate**: $5-15 (when running)
- **Lambda**: $1-5
- **S3**: $1-10 (depending on video storage)
- **CloudFront**: $1-5
- **Other services**: $2-5

**Total estimated**: $25-65/month

### Cost Optimization Tips
- Use `demo` environment for testing
- Delete resources when not needed
- Monitor usage in AWS Cost Explorer
- Set up billing alerts

## 🔄 Updating Your Deployment

### Automatic Updates
- Push changes to `main` branch
- Workflows trigger automatically for infrastructure changes

### Manual Updates
- Run the deployment workflow again
- Only changed resources will be updated
- Zero-downtime updates for most changes

## 🎉 Next Steps

After deployment:

1. **Test the application**
   - Upload a sample video
   - Try different search queries
   - Check video processing pipeline

2. **Customize the frontend**
   - Modify React components
   - Update styling and branding
   - Add new features

3. **Integrate with your systems**
   - Use the API endpoints
   - Build custom applications
   - Set up monitoring and alerts

4. **Scale for production**
   - Increase ECS task counts
   - Add more OpenSearch nodes
   - Set up proper monitoring

## 📚 Additional Resources

- [AWS CloudFormation Documentation](https://docs.aws.amazon.com/cloudformation/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)
- [OpenSearch Service Pricing](https://aws.amazon.com/opensearch-service/pricing/)

---

**Happy deploying! 🚀**

If you run into any issues, please create an issue in this repository with details about your deployment configuration and error messages.