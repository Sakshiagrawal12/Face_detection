// Jenkinsfile.ecr-simple
pipeline {
    agent any
    
    environment {
        AWS_ACCOUNT_ID = '693149914819'   // ❗ FIXED (no hyphens)
        AWS_REGION = 'ap-south-1'
        IMAGE_NAME = 'mask-detector'
        ECR_REPO = "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}"
        DEPLOY_SERVER_IP = '12345'
        DOCKER_PATH = 'C:\ProgramData\Microsoft\Windows\Start Menu'   // ✅ Added
    }
    
    stages {

        stage('Debug Environment') {
            steps {
                sh """
                    echo "PATH = $PATH"
                    which docker || true
                    ${DOCKER_PATH} --version || true
                """
            }
        }

        stage('Checkout') {
            steps {
                git url: 'https://github.com/Bharadwaj-8/Face_detection.git', branch: 'feature'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    sh """
                        ${DOCKER_PATH} build -t ${IMAGE_NAME}:latest .
                        ${DOCKER_PATH} tag ${IMAGE_NAME}:latest ${ECR_REPO}:latest
                    """
                }
            }
        }

        stage('Login to ECR') {
            steps {
                script {
                    withAWS(credentials: 'aws-credentials', region: "${AWS_REGION}") {
                        sh """
                            aws ecr get-login-password --region ${AWS_REGION} | ${DOCKER_PATH} login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
                        """
                    }
                }
            }
        }

        stage('Push to ECR') {
            steps {
                sh "${DOCKER_PATH} push ${ECR_REPO}:latest"
            }
        }

        stage('Deploy to EC2') {
            steps {
                sh """
                    ssh ec2-user@${DEPLOY_SERVER_IP} "
                        aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com &&
                        docker pull ${ECR_REPO}:latest &&
                        docker stop mask-detector || true &&
                        docker rm mask-detector || true &&
                        docker run -d --name mask-detector -p 8081:8081 ${ECR_REPO}:latest
                    "
                """
            }
        }
    }
}
