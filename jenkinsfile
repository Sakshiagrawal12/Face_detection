// Jenkinsfile.ecr-simple
pipeline {
    agent any
    
    environment {
        AWS_ACCOUNT_ID = '6931-4991-4819'
        AWS_REGION = 'ap-south-1'
        ECR_REPO = "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/mask-detector"
        DEPLOY_SERVER_IP = '13.201.94.72'
    }
    
    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/Sakshiagrawal12/Face_detection.git'
            }
        }
        
        stage('Build') {
            steps {
                sh 'docker build -t mask-detector:latest .'
                sh "docker tag mask-detector:latest ${ECR_REPO}:latest"
            }
        }
        
        stage('Push to ECR') {
            steps {
                withAWS(credentials: 'aws-credentials', region: "${AWS_REGION}") {
                    sh """
                        aws ecr get-login-password | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
                        docker push ${ECR_REPO}:latest
                    """
                }
            }
        }
        
        stage('Deploy') {
            steps {
                sh """
                    ssh ec2-user@${DEPLOY_SERVER_IP} '
                        aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
                        docker pull ${ECR_REPO}:latest
                        docker stop mask-detector || true
                        docker rm mask-detector || true
                        docker run -d --name mask-detector -p 8080:8000 ${ECR_REPO}:latest
                    '
                """
            }
        }
    }
}