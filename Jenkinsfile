// Jenkinsfile.ecr-simple
pipeline {
    agent any
    
    environment {
        AWS_ACCOUNT_ID = '693149914819'   // ❗ FIXED (no hyphens)
        AWS_REGION = 'ap-south-1'
        IMAGE_NAME = 'mask-detector'
        ECR_REPO = "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}"
        DEPLOY_SERVER_IP = '12345'
        DOCKER_PATH = 'C:/Program Files/Docker/Docker/resources/bin'   // ✅ Added
    }
    
   stages {

    stage('Debug Environment') {
        steps {
            bat '''
                echo PATH = %PATH%
                docker --version
            '''
        }
    }

    stage('Checkout') {
        steps {
            git url: 'https://github.com/Bharadwaj-8/Face_detection.git', branch: 'feature-automation2'
        }
    }

    stage('Build Docker Image') {
        steps {
            bat '''
                docker build -t %IMAGE_NAME%:latest .
                docker tag %IMAGE_NAME%:latest %ECR_REPO%:latest
            '''
        }
    }

    stage('Login to ECR') {
        steps {
            withAWS(credentials: 'aws-credentials', region: "%AWS_REGION%") {
                bat '''
                    aws ecr get-login-password --region %AWS_REGION% | docker login --username AWS --password-stdin %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com
                '''
            }
        }
    }

    stage('Push to ECR') {
        steps {
            bat '''
                docker push %ECR_REPO%:latest
            '''
        }
    }

    stage('Deploy to EC2') {
        steps {
            bat '''
                ssh ec2-user@%DEPLOY_SERVER_IP% ^
                "aws ecr get-login-password --region %AWS_REGION% | docker login --username AWS --password-stdin %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com && ^
                docker pull %ECR_REPO%:latest && ^
                docker stop mask-detector || echo not running && ^
                docker rm mask-detector || echo not exists && ^
                docker run -d --name mask-detector -p 8081:8081 %ECR_REPO%:latest"
            '''
        }
    }
}
